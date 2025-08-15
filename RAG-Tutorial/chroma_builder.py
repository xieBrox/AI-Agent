import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import hashlib
import uuid

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_builder.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChromaBuilder")

class ChromaKnowledgeBase:
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "default"):
        """
        Initialize Chroma knowledge base
        
        Args:
            persist_directory: Database persistence directory
            embedding_model: Embedding model type ("default", "sentence-transformers")
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Select embedding function
        if embedding_model == "default":
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            logger.info("Using ChromaDB default embedding function")
        elif embedding_model == "sentence-transformers":
            try:
                # Use lightweight Chinese multilingual model
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
                logger.info("Using SentenceTransformer embedding function")
            except Exception as e:
                logger.warning(f"SentenceTransformer initialization failed, using default function: {e}")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            logger.info("Using ChromaDB default embedding function")
        
        # Create main collection
        self.main_collection = self.client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_function,
            metadata={"description": "Main knowledge base collection"}
        )
        
        logger.info(f"Chroma knowledge base initialized successfully, data directory: {persist_directory}")

    def _generate_unique_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID"""
        # Generate unique identifier using text content and source file information
        source_info = f"{metadata.get('source', '')}-{metadata.get('chunk_id', 0)}"
        unique_string = f"{source_info}-{text[:100]}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def load_from_jsonl(self, jsonl_file: str, collection_name: str = "knowledge_base", 
                       batch_size: int = 100) -> int:
        """
        Batch load data from JSONL file to Chroma database
        
        Args:
            jsonl_file: Path to JSONL file output by document_processor.py
            collection_name: Name of the collection
            batch_size: Batch processing size
            
        Returns:
            Number of successfully loaded documents
        """
        jsonl_path = Path(jsonl_file)
        if not jsonl_path.exists():
            logger.error(f"JSONL file does not exist: {jsonl_file}")
            return 0

        # Get or create collection
        if collection_name == "knowledge_base":
            collection = self.main_collection
        else:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

        # Read JSONL data
        documents = []
        metadatas = []
        ids = []
        
        logger.info(f"Starting to load data from {jsonl_file}...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '').strip()
                    metadata = data.get('metadata', {})
                    
                    if not text:
                        logger.warning(f"Line {line_num} has empty text, skipping")
                        continue
                    
                    # Generate unique ID
                    doc_id = self._generate_unique_id(text, metadata)
                    
                    # Add additional metadata
                    enhanced_metadata = {
                        **metadata,
                        "load_timestamp": str(Path(jsonl_file).stat().st_mtime),
                        "line_number": line_num,
                        "collection": collection_name
                    }
                    
                    documents.append(text)
                    metadatas.append(enhanced_metadata)
                    ids.append(doc_id)
                    
                    # Batch processing
                    if len(documents) >= batch_size:
                        self._add_batch_to_collection(collection, documents, metadatas, ids)
                        documents, metadatas, ids = [], [], []
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")

        # Process remaining documents
        if documents:
            self._add_batch_to_collection(collection, documents, metadatas, ids)
        
        total_count = collection.count()
        logger.info(f"Data loading completed, collection '{collection_name}' contains {total_count} documents")
        return total_count

    def _add_batch_to_collection(self, collection, documents: List[str], 
                               metadatas: List[Dict], ids: List[str]):
        """Batch add documents to collection"""
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.debug(f"Successfully added {len(documents)} documents to collection")
        except Exception as e:
            logger.error(f"Failed to add batch of documents: {e}")
            # Try adding individually to identify problematic documents
            for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
                try:
                    collection.add(
                        documents=[doc],
                        metadatas=[meta],
                        ids=[doc_id]
                    )
                except Exception as single_error:
                    logger.error(f"Failed to add individual document (index {i}): {single_error}")

    def search_knowledge(self, query: str, collection_name: str = "knowledge_base",
                        n_results: int = 5, filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Search the knowledge base
        
        Args:
            query: Query text
            collection_name: Name of the collection
            n_results: Number of results to return
            filter_metadata: Metadata filter conditions
            
        Returns:
            Search results
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            search_params = {
                "query_texts": [query],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if filter_metadata:
                search_params["where"] = filter_metadata
            
            results = collection.query(** search_params)
            
            logger.info(f"Query '{query[:50]}...' returned {len(results['documents'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        try:
            collections = self.client.list_collections()
            
            for collection in collections:
                try:
                    count = collection.count()
                    
                    # Get sample document
                    sample = collection.peek(limit=1)
                    sample_metadata = sample['metadatas'][0] if sample['metadatas'] else {}
                    
                    stats[collection.name] = {
                        "name": collection.name,
                        "document_count": count,
                        "sample_metadata_keys": list(sample_metadata.keys()) if sample_metadata else [],
                        "status": "Normal"
                    }
                except Exception as e:
                    stats[collection.name] = {
                        "name": collection.name,
                        "document_count": 0,
                        "error": str(e),
                        "status": "Error"
                    }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def create_specialized_collection(self, collection_name: str, 
                                    description: str = "") -> bool:
        """Create a specialized collection"""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": description}
            )
            logger.info(f"Created collection '{collection_name}': {description}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def export_collection_data(self, collection_name: str, output_file: str) -> bool:
        """Export collection data in JSONL format"""
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get all data
            all_data = collection.get(include=["documents", "metadatas", "embeddings"])
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, (doc, meta) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                    export_item = {
                        "text": doc,
                        "metadata": meta,
                        "id": all_data['ids'][i]
                    }
                    f.write(json.dumps(export_item, ensure_ascii=False) + '\n')
            
            logger.info(f"Collection '{collection_name}' data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            collections = self.client.list_collections()
            logger.info(f"✅ Chroma database connection successful, total {len(collections)} collections")
            
            for collection in collections:
                try:
                    count = collection.count()
                    logger.info(f"  📚 {collection.name}: {count} documents")
                except Exception as e:
                    logger.warning(f"  ⚠️ {collection.name}: Unable to get count ({e})")
            
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

    def quick_test_search(self, query: str = "test") -> bool:
        """Quick test of search functionality"""
        try:
            stats = self.get_collection_stats()
            for name, info in stats.items():
                if info['document_count'] > 0:
                    logger.info(f"🔍 Testing search in collection '{name}': '{query}'")
                    results = self.search_knowledge(query, collection_name=name, n_results=2)
                    
                    if results['documents'][0]:
                        logger.info("✅ Search test successful!")
                        for i, doc in enumerate(results['documents'][0]):
                            logger.info(f"  Result {i+1}: {doc[:80]}...")
                        return True
                    else:
                        logger.info(f"⚠️ No results found in collection '{name}', trying next collection")
            
            logger.warning("❌ No results found in any collection")
            return False
            
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")
            return False

# Batch processing function
def build_knowledge_base_from_processed_data(
    processed_data_dir: str = "./processed_data",
    chroma_db_dir: str = "./chroma_db",
    embedding_model: str = "default"
) -> ChromaKnowledgeBase:
    """
    Build knowledge base from output of document_processor.py
    
    Args:
        processed_data_dir: Directory containing processed data
        chroma_db_dir: Chroma database directory
        embedding_model: Embedding model type
        
    Returns:
        Built knowledge base instance
    """
    # Initialize knowledge base
    kb = ChromaKnowledgeBase(persist_directory=chroma_db_dir, embedding_model=embedding_model)
    
    # Find all JSONL files
    data_dir = Path(processed_data_dir)
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {processed_data_dir}")
        return kb
    
    logger.info(f"Found {len(jsonl_files)} JSONL files")
    
    # Process files one by one
    total_docs = 0
    for jsonl_file in tqdm(jsonl_files, desc="Building knowledge base"):
        # Create collection name based on filename
        collection_name = jsonl_file.stem.replace("processed_", "").replace("_data", "")
        if not collection_name or collection_name == "processed":
            collection_name = "knowledge_base"
        
        docs_count = kb.load_from_jsonl(str(jsonl_file), collection_name=collection_name)
        total_docs += docs_count
    
    logger.info(f"🎉 Knowledge base build completed! Total {total_docs} documents loaded")
    
    # Display statistics
    stats = kb.get_collection_stats()
    logger.info("📊 Collection statistics:")
    for name, info in stats.items():
        logger.info(f"  {name}: {info['document_count']} documents")
    
    return kb

if __name__ == "__main__":
    # Example usage
    logger.info("🚀 Starting to build Chroma knowledge base...")
    
    # Method 1: Batch build from processed data
    kb = build_knowledge_base_from_processed_data(
        processed_data_dir="./processed_data",
        chroma_db_dir="./chroma_db",
        embedding_model="default"  # or "sentence-transformers"
    )
    
    # Test connection
    kb.test_connection()
    
    # Test search - intelligently select collection with data
    stats = kb.get_collection_stats()
    if stats:
        # Find collection with data for testing
        test_collection = None
        for name, info in stats.items():
            if info['document_count'] > 0:
                test_collection = name
                break
        
        if test_collection:
            test_query = "Witch gameplay"  # Translated from "女巫玩法"
            logger.info(f"🔍 Testing search: {test_query} (in collection: {test_collection})")
            results = kb.search_knowledge(test_query, collection_name=test_collection, n_results=3)
            
            if results['documents'][0]:
                logger.info("Search results:")
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    logger.info(f"  Result {i+1} (similarity: {1-distance:.3f}): {doc[:100]}...")
                    logger.info(f"    Source: {metadata.get('source', 'Unknown')}")
            else:
                logger.info("No relevant results found")
        else:
            logger.info("All collections are empty, skipping search test")
    
    logger.info("✅ Knowledge base build and testing completed!")
