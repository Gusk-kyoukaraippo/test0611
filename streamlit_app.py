import streamlit as st
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.openai import OpenAIEmbedding # ★ここをOpenAIEmbeddingに変更
from typing import List, Dict, Any, Set
import sys
import os
import re

# --- Streamlit アプリケーションのメインロジック ---
st.set_page_config(page_title="RAGベースの質問検索システム", layout="wide")

st.title("📚 RAGベースの質問検索システム")
st.markdown("このシステムは、北海道と沖縄を答えとする質問が200入っています。<br>"
            "あなたの入力に基づいて最も関連性の高い**質問**を検索し、質問と回答を提示します。")

# JavaScriptでページを一番上までスクロールする関数 (st.session_state と連携)
def scroll_to_top_conditional():
    if st.session_state.get('should_scroll_to_top', False):
        js_code = """
        <script>
            window.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """
        st.components.v1.html(js_code, height=0, width=0)
        st.session_state.should_scroll_to_top = False # スクロール実行後、フラグをリセット

# アプリのレンダリングサイクルの早い段階でスクロールチェック関数を呼び出す
# これにより、UIの更新前にスクロールがトリガーされる可能性が高まる
scroll_to_top_conditional()

# --- バックエンドのRAGSystemクラスをStreamlitアプリに統合 ---
class RAGSystem:
    """
    RAG (Retrieval Augmented Generation) システムを実装するクラス。
    CSVファイルからデータを読み込み、ベクトル検索とBM25検索を組み合わせて情報を取得し、
    Gemini LLMを用いて最終的な回答を生成します。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        RAGSystemのインスタンスを初期化します。
        """
        self.config = config
        self.documents: List[Document] = []
        self.document_map: Dict[str, Document] = {} 
        self.vector_retriever: BaseRetriever = None
        self.bm25_retriever: BM25Retriever = None
        self.response_synthesizer: Any = None
        self.llm: GoogleGenAI = None
        self.embed_model: OpenAIEmbedding = None # OpenAIEmbeddingを追加

        self._initialize_system()

    def _initialize_system(self):
        """
        RAGシステムを初期化し、必要なコンポーネントをセットアップします。
        """
        gemini_api_key = "GEMINI_API_KEY" # あなたのAPIキーに置き換えてください
        openai_api_key = "OPENAI_API_KEY"
        self.config["gemini_api_key"] = gemini_api_key
        self.config["openai_api_key"] = openai_api_key


        self.documents = self._load_documents_from_csv(self.config["csv_file_path"])
        
        self.document_map = {doc.id_: doc for doc in self.documents}

        if not self.documents:
            st.error("ドキュメントが読み込まれていないため、システム初期化を中止します。")
            st.stop()
            
        

        # --- OpenAI text-embedding-3-small を埋め込みモデルとして設定 ---
        # model_name='text-embedding-3-small' を指定します
        self.embed_model = OpenAIEmbedding(
            api_key=openai_api_key,
            model_name='text-embedding-3-small' 
        )
        # --- 埋め込みモデルをVectorStoreIndexに渡す ---
        vector_index = VectorStoreIndex.from_documents(
            self.documents,
            embed_model=self.embed_model # ここでOpenAIの埋め込みモデルを指定
        )
        self.vector_retriever = vector_index.as_retriever(similarity_top_k=self.config["vector_top_k"])

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.documents,
            similarity_top_k=self.config["bm25_top_k"]
        )

        self.llm = GoogleGenAI(
            api_key=self.config["gemini_api_key"],
            model=self.config["gemini_model_name"]
        )
        
        self.response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact"
        )

    def _load_documents_from_csv(self, file_path: str) -> List[Document]:
        """
        指定されたCSVファイルからドキュメントを読み込み、LlamaIndexのDocumentオブジェクトのリストを返します。
        """
        documents: List[Document] = []
        if not os.path.exists(file_path):
            st.error(f"エラー: '{file_path}' が見つかりません。ファイルパスを確認してください。")
            st.stop()

        try:
            df = pd.read_csv(file_path)
            required_columns = ['id', 'question', 'answer', 'image_path']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                st.error(f"エラー: CSVに必要なカラムが見つかりません。欠けているカラム: {', '.join(missing_cols)}")
                st.stop()

            for _, row in df.iterrows():
                doc = Document(
                    text=row['question'],
                    metadata={
                        "answer": row['answer'],
                        "image": row['image_path']
                    },
                    id_=str(row['id'])
                )
                documents.append(doc)
        except pd.errors.EmptyDataError:
            st.warning(f"警告: '{file_path}' は空のファイルです。ドキュメントは読み込まれませんでした。")
        except Exception as e:
            st.error(f"CSVファイルの読み込み中に予期せぬエラーが発生しました: {e}")
            st.stop()
        return documents

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        与えられたクエリに対して検索を実行し、LLMに最適な質問を選ばせます。
        LLMが質問を選びきれなかった場合は、検索結果の最初の（最もスコアの高い）質問を返します。
        """
        query_bundle = QueryBundle(query_str=query_text)

        with st.spinner("情報を検索中..."):
            vector_nodes = self.vector_retriever.retrieve(query_bundle)
            bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        combined_nodes: List[NodeWithScore] = []
        seen_node_ids: Set[str] = set()

        all_retrieved_nodes_unsorted = vector_nodes + bm25_nodes
        all_retrieved_nodes_unsorted.sort(key=lambda x: x.score if x.score is not None else -1, reverse=True)

        for node in all_retrieved_nodes_unsorted:
            if node.node_id not in seen_node_ids:
                combined_nodes.append(node)
                seen_node_ids.add(node.node_id)
        
        if not combined_nodes:
            st.warning("関連する質問が見つかりませんでした。")
            return {
                "selected_question_id": None,
                "retrieved_nodes": [],
                "llm_response_text": "関連する質問が見つかりませんでした。"
            }

        context_str = ""
        for i, node in enumerate(combined_nodes):
            context_str += f"--- 質問 {i+1} (ID: {node.node_id}) ---\n"
            context_str += f"質問内容: {node.get_content()}\n"
            context_str += f"関連する回答: {node.metadata.get('answer', 'N/A')}\n\n"

        llm_query_prompt = f"""
以下の検索結果の中から、ユーザーのクエリに最も関連性の高い「質問」のIDを1つだけ選んでください。
必ず、最も妥当だと思われる質問のIDを1つ回答してください。
選択肢がない場合でも、無理にでも1つ選んでください。
回答は、選んだ質問のIDのみを正確に記述してください。ID以外は一切含めないでください。

--- ユーザーのクエリ ---
{query_text}

--- 検索結果 ---
{context_str}

--- あなたの回答 (選んだ質問のID、例: "123" または "71e4f049-0fe7-4766-9c68-2c11dcc97e95") ---
"""
        with st.spinner("LLMが最適な質問を選択中..."):
            llm_response = self.llm.complete(llm_query_prompt)
        
        raw_llm_response_text = str(llm_response).strip()
        
        uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
        match = re.search(uuid_pattern, raw_llm_response_text)
        
        if not match:
            match = re.search(r'\d+', raw_llm_response_text)

        selected_id = match.group(0) if match else None

        combined_node_ids = {node.node_id for node in combined_nodes}

        if selected_id is None or selected_id not in combined_node_ids:
            st.warning("LLMは有効な質問IDを選びませんでした。検索結果の最初の質問をデフォルトとして選択します。")
            selected_id = combined_nodes[0].node_id
        
        return {
            "selected_question_id": selected_id,
            "retrieved_nodes": combined_nodes,
            "llm_response_text": raw_llm_response_text
        }


# --- 設定値 ---
CONFIG: Dict[str, Any] = {
    "csv_file_path": 'dummy2.csv', 
    "vector_top_k": 3, 
    "bm25_top_k": 3, 
    "gemini_model_name": "gemini-2.0-flash" 
}

@st.cache_resource
def get_rag_system():
    try:
        rag_system = RAGSystem(CONFIG)
        return rag_system
    except SystemExit:
        st.error("RAGシステムの初期化中に致命的なエラーが発生しました。ファイルパスやAPIキーを確認してください。")
        st.stop()
    except Exception as e:
        st.error(f"RAGシステムの初期化中にエラーが発生しました: {e}")
        st.stop()

rag_system = get_rag_system()

st.divider()

# st.session_state の初期化
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_input_section' not in st.session_state:
    st.session_state.show_input_section = True 
if 'last_result' not in st.session_state:
    st.session_state.last_result = None 
if 'display_node_id' not in st.session_state:
    st.session_state.display_node_id = None 
if 'should_scroll_to_top' not in st.session_state:
    st.session_state.should_scroll_to_top = False 
if 'user_entered_query' not in st.session_state: # 新しく追加
    st.session_state.user_entered_query = ""

# 新しい検索を開始する関数
def reset_search():
    st.session_state.current_query = ""
    st.session_state.show_input_section = True
    st.session_state.last_result = None 
    st.session_state.display_node_id = None 
    st.session_state.should_scroll_to_top = True 
    st.session_state.user_entered_query = "" # リセット時にもクリア
    st.rerun()

# --- 入力セクション ---
if st.session_state.show_input_section:
    col_input, col_button = st.columns([0.8, 0.2])
    with col_input:
        user_query_input = st.text_input(
            "質問を入力してください：",
            value=st.session_state.current_query, 
            placeholder="例: 「寒いのは？」",
            key="query_input"
        )
    with col_button:
        st.write("") 
        st.write("") 
        search_button = st.button("検索", key="search_button")

    # ユーザーが入力した内容を保存
    if user_query_input:
        st.session_state.user_entered_query = user_query_input

    if (search_button and user_query_input) or \
       (user_query_input and user_query_input != st.session_state.current_query and st.session_state.get('last_query_input', '') != user_query_input):
        
        st.session_state.current_query = user_query_input
        st.session_state.show_input_section = False 
        st.session_state.last_query_input = user_query_input 
        st.session_state.last_result = None 
        st.session_state.display_node_id = None 
        st.session_state.should_scroll_to_top = True 
        st.rerun() 
    elif search_button and not user_query_input:
        st.warning("質問を入力してから検索ボタンを押してください。")

# --- 検索処理と結果の保存 ---
# current_query があり、かつ入力セクションが非表示の場合に検索処理を実行
# また、まだ検索結果が保存されていない場合にのみ実行
if st.session_state.current_query and not st.session_state.show_input_section and st.session_state.last_result is None:
    # ここでユーザーの入力を表示
    st.write(f"あなたの質問: **{st.session_state.user_entered_query}**") # ここを修正
    
    with st.spinner("情報を検索中..."):
        try:
            result = rag_system.query(st.session_state.current_query)
            st.session_state.last_result = result 
            st.session_state.display_node_id = result["selected_question_id"] 
            st.session_state.should_scroll_to_top = True 
            st.rerun() 
        except Exception as e:
            st.error(f"クエリ処理中にエラーが発生しました: {e}")
            st.session_state.last_result = {"selected_question_id": None, "retrieved_nodes": [], "llm_response_text": f"エラー: {e}"}
            st.session_state.show_input_section = True 
            st.session_state.should_scroll_to_top = True 
            st.rerun()

# --- メイン表示エリアのロジック ---
if st.session_state.last_result and st.session_state.display_node_id:
    target_node = next((node for node in st.session_state.last_result["retrieved_nodes"] if node.node_id == st.session_state.display_node_id), None)

    if target_node:
        st.markdown(f"**あなたの質問:** `{st.session_state.current_query}`") # または user_entered_query
        st.success("✨ 質問と回答") 
        st.markdown(f"**検索された質問内容:** \n{target_node.get_content()}")
        st.markdown(f"**回答:** \n{target_node.metadata.get('answer', 'N/A')}")
        
        image_path = target_node.metadata.get('image', None)
        if image_path and os.path.exists(image_path):
            try:
                st.image(image_path, caption="関連画像", width=300)
            except Exception as e:
                st.warning(f"画像の表示中にエラーが発生しました: {e}")
        elif image_path:
            st.warning(f"画像ファイル '{image_path}' が見つかりませんでした。")
    else:
        st.warning(f"ID `{st.session_state.display_node_id}` に対応する質問が見つかりませんでした。")

    st.divider()

    # --- その他の検索結果 ---
    with st.expander("🔍 その他の検索結果を見る"):
        if st.session_state.last_result and st.session_state.last_result["retrieved_nodes"]:
            for i, node in enumerate(st.session_state.last_result["retrieved_nodes"]):
                if node.node_id == st.session_state.display_node_id:
                    st.markdown(f"**--- 現在表示中の質問 (ID: `{node.node_id}`) ---**")
                    st.write(f"質問: {node.get_content()[:100]}...")
                    st.markdown("---")
                    continue

                st.markdown(f"**--- 検索結果 {i+1} (ID: `{node.node_id}`) ---**")
                st.write(f"**質問:** {node.get_content()}")
                st.write(f"**回答の冒頭:** {node.metadata.get('answer', 'N/A')[:100]}...") 
                if node.score is not None:
                    st.write(f"**スコア:** {node.score:.4f}")
                
                # この質問を表示するためのボタン
                if st.button(f"この質問を表示 (ID: {node.node_id})", key=f"select_node_{node.node_id}"):
                    st.session_state.display_node_id = node.node_id 
                    st.session_state.should_scroll_to_top = True 
                    st.rerun() 
                st.markdown("---")
        else:
            st.info("表示するその他の検索結果はありません。")
            
    st.divider()
    # 新しい検索を開始するボタン
    st.button("新しい検索を開始する", on_click=reset_search)

# --- 検索結果がなく、入力セクションも表示されていない場合は、エラーまたは完了状態 ---
elif not st.session_state.show_input_section and not st.session_state.current_query and not st.session_state.last_result:
    st.warning("検索結果が見つかりませんでした。")
    st.button("新しい検索を開始する", on_click=reset_search)


st.caption("powered by LlamaIndex & Google Gemini")