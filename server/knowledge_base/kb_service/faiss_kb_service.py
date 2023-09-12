import os
import shutil

from configs.model_config import (
    KB_ROOT_PATH,
    CACHED_VS_NUM,
    EMBEDDING_MODEL,
    SCORE_THRESHOLD
)
from server.knowledge_base.kb_service.base import KBService, SupportedVSType
from functools import lru_cache
from server.knowledge_base.utils import get_vs_path, load_embeddings, KnowledgeFile
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Optional
from langchain.docstore.document import Document
from server.utils import torch_gc, embedding_device


_VECTOR_STORE_TICKS = {}


@lru_cache(CACHED_VS_NUM)
def load_faiss_vector_store(
        knowledge_base_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
        embeddings: Embeddings = None,
        tick: int = 0,  # tick will be changed by upload_doc etc. and make cache refreshed.
) -> FAISS:
    print(f"loading vector store in '{knowledge_base_name}'.")
    vs_path = get_vs_path(knowledge_base_name)
    if embeddings is None:
        embeddings = load_embeddings(embed_model, embed_device)

    if not os.path.exists(vs_path):
        os.makedirs(vs_path)

    if "index.faiss" in os.listdir(vs_path):
        search_index = FAISS.load_local(vs_path, embeddings, normalize_L2=True)
    else:
        # create an empty vector store
        '''
1. 首先，创建一个新的`Document`实例`doc`，其中的`page_content`值为`"init"`，并没有元数据（`metadata`）。

2. 接着，使用FAISS的`from_documents`函数创建一个新的FAISS索引`search_index`。这个函数接收一个文档列表（在这里只有一个文档`doc`），一个嵌入（`embeddings`），和一个指示是否对嵌入进行L2标准化的参数（在这里设为`True`）。

3. 然后，获取`search_index`中所有文档的ID。这里，`_dict`应该是`docstore`对象的一个属性，其包含所有文档的字典，键是文档ID，值是文档内容。因此，`[k for k, v in search_index.docstore._dict.items()]`表示获取所有文档的ID。

4. 使用FAISS索引的`delete`函数删除所有这些文档。虽然我们刚创建了一个含有一个文档的FAISS索引，但是我们马上删除这个文档，从而得到一个空的FAISS索引。

5. 最后，使用FAISS索引的`save_local`函数将这个空的FAISS索引保存到本地路径`vs_path`。

总的来说，这段代码创建了一个空的FAISS索引并保存到了本地。这个索引可以后续被用来存储文档的向量嵌入，并进行相似性搜索。
        '''
        doc = Document(page_content="init", metadata={})
        search_index = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        """
            db = FAISS.from_documents(docs, embeddings) 是使用 FAISS 库创建索引的代码。它将文档和对应的嵌入向量作为输入，
            并返回一个 FAISS 索引对象，该对象可以用于快速检索相似的文档。
            db = FAISS.from_documents(docs, embeddings)只是在内存中创建了一个 FAISS 索引对象，用于快速检索相似的文档。如果需要将索引保存到磁盘上，
            db.save_local("faiss_index") 将 FAISS 索引对象保存到本地文件系统
            在代码中db = FAISS.from_documents(docs, embeddings)，docs首先使用embeddings函数将它们嵌入到向量表示中，然后传递给类from_documents的方法FAISS以创建 FAISS 索引。
            该embeddings函数通常是一个预先训练的模型，它将每个文档映射docs到高维向量表示。然后使用该向量表示来构建 FAISS 索引，该索引允许基于向量表示对文档进行高效的相似性搜索和检索。
        """
        ids = [k for k, v in search_index.docstore._dict.items()]
        search_index.delete(ids)
        search_index.save_local(vs_path)
    
    if tick == 0: # vector store is loaded first time
        _VECTOR_STORE_TICKS[knowledge_base_name] = 0

    return search_index


def refresh_vs_cache(kb_name: str):
    """
    make vector store cache refreshed when next loading
    """
    _VECTOR_STORE_TICKS[kb_name] = _VECTOR_STORE_TICKS.get(kb_name, 0) + 1
    print(f"知识库 {kb_name} 缓存刷新：{_VECTOR_STORE_TICKS[kb_name]}")


class FaissKBService(KBService):
    vs_path: str
    kb_path: str

    def vs_type(self) -> str:
        return SupportedVSType.FAISS

    def get_vs_path(self):
        return os.path.join(self.get_kb_path(), "vector_store")

    def get_kb_path(self):
        return os.path.join(KB_ROOT_PATH, self.kb_name)

    def load_vector_store(self) -> FAISS:
        return load_faiss_vector_store(
            knowledge_base_name=self.kb_name,
            embed_model=self.embed_model,
            tick=_VECTOR_STORE_TICKS.get(self.kb_name, 0),
        )

    def save_vector_store(self, vector_store: FAISS = None):
        vector_store = vector_store or self.load_vector_store()
        vector_store.save_local(self.vs_path)
        return vector_store

    def refresh_vs_cache(self):
        refresh_vs_cache(self.kb_name)

    def get_doc_by_id(self, id: str) -> Optional[Document]:
        vector_store = self.load_vector_store()
        return vector_store.docstore._dict.get(id)

    def do_init(self):
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()

    def do_create_kb(self):
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        self.load_vector_store()

    def do_drop_kb(self):
        self.clear_vs()
        shutil.rmtree(self.kb_path)

    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float = SCORE_THRESHOLD,
                  embeddings: Embeddings = None,
                  ) -> List[Document]:
        search_index = self.load_vector_store()
        docs = search_index.similarity_search_with_score(query, k=top_k, score_threshold=score_threshold)
        return docs
    """
    将切片后的文档信息存储到FAISS
    """
    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        # 创建空的FAISS索引并保存到了本地， <langchain.vectorstores.faiss.FAISS object at 0x7ff0ff2b8940>
        vector_store = self.load_vector_store()

        # `add_documents`方法可能会将这些`Document`对象添加到`vector_store`中，并返回一个与`docs`中每个对象对应的id列表。这个id列表被赋值给了变量`ids`。
        #  docs: [Document(page_content='睡前故事小狗汪汪历..', metadata={'source': '/usr/xxtcode/chatgpt/Langchain-Chatchat/knowledge_base/samples/content/1.pdf'})]
        # ids： ['ab7f9715-2561-4da0-ad75-73ac8d0cfe09', 'c9a8c50f-cfae-4ca1-a4fd-47c115f10b7d'...]
        ids = vector_store.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        torch_gc() #这个函数用于清空 PyTorch 在 GPU 上分配的缓存，以释放 GPU 内存
        if not kwargs.get("not_refresh_vs_cache"):
            vector_store.save_local(self.vs_path)
            self.refresh_vs_cache()
        return doc_infos

    def do_delete_doc(self,
                      kb_file: KnowledgeFile,
                      **kwargs):
        vector_store = self.load_vector_store()

        ids = [k for k, v in vector_store.docstore._dict.items() if v.metadata["source"] == kb_file.filepath]
        if len(ids) == 0:
            return None

        vector_store.delete(ids)
        if not kwargs.get("not_refresh_vs_cache"):
            vector_store.save_local(self.vs_path)
            self.refresh_vs_cache()

        return vector_store

    def do_clear_vs(self):
        shutil.rmtree(self.vs_path)
        os.makedirs(self.vs_path)
        self.refresh_vs_cache()

    def exist_doc(self, file_name: str):
        if super().exist_doc(file_name):
            return "in_db"

        content_path = os.path.join(self.kb_path, "content")
        if os.path.isfile(os.path.join(content_path, file_name)):
            return "in_folder"
        else:
            return False


if __name__ == '__main__':
    faissService = FaissKBService("test")
    faissService.add_doc(KnowledgeFile("README.md", "test"))
    faissService.delete_doc(KnowledgeFile("README.md", "test"))
    faissService.do_drop_kb()
    print(faissService.search_docs("如何启动api服务"))