import chromadb
from chromadb.utils import embedding_functions
import json
import os

class KnowledgeBase:
    def __init__(self, persist_directory="medical_kb"):
        """
        Initialize the knowledge base, using a lightweight embedding function to avoid network download issues
        """
        # Using the new ChromaDB 1.x API
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Using ChromaDB's built-in default embedding function to avoid downloading large models
        # This function uses simple sentence embeddings and does not require downloading additional models
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        
        # Create different types of knowledge base collections with specified embedding function
        self.symptoms_collection = self.client.get_or_create_collection(
            name="symptoms",
            embedding_function=default_ef
        )
        self.diseases_collection = self.client.get_or_create_collection(
            name="diseases", 
            embedding_function=default_ef
        )
        self.treatments_collection = self.client.get_or_create_collection(
            name="treatments",
            embedding_function=default_ef
        )
    
    def add_medical_knowledge(self, collection_name: str, documents: list, metadatas: list, ids: list):
        """Add medical knowledge to the specified collection"""
        try:
            if collection_name == "symptoms":
                collection = self.symptoms_collection
            elif collection_name == "diseases":
                collection = self.diseases_collection
            elif collection_name == "treatments":
                collection = self.treatments_collection
            else:
                # Create new collection for new collection names
                default_ef = embedding_functions.DefaultEmbeddingFunction()
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=default_ef
                )
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Successfully added {len(documents)} records to the {collection_name} collection")
            
        except Exception as e:
            print(f"❌ Failed to add medical knowledge: {str(e)}")
            raise
    
    def search_knowledge(self, collection_name: str, query: str, n_results: int = 5):
        """Search for relevant medical knowledge"""
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Failed to search knowledge: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

    def get_disease_info(self, symptoms: list) -> dict:
        """Query possible disease information based on symptoms"""
        if not symptoms:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
            
        try:
            query = " ".join(symptoms)
            results = self.diseases_collection.query(
                query_texts=[query],
                n_results=3
            )
            return results
        except Exception as e:
            print(f"❌ Failed to query disease information: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

    def get_treatment_suggestions(self, disease: str) -> dict:
        """Get treatment suggestions based on the disease"""
        if not disease:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
            
        try:
            results = self.treatments_collection.query(
                query_texts=[disease],
                n_results=3
            )
            return results
        except Exception as e:
            print(f"❌ Failed to query treatment suggestions: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
    
    def get_collection_info(self):
        """Get basic information for all collections"""
        try:
            collections = self.client.list_collections()
            info = {}
            for collection in collections:
                try:
                    count = collection.count()
                    info[collection.name] = {
                        "name": collection.name,
                        "count": count
                    }
                except Exception as e:
                    info[collection.name] = {
                        "name": collection.name,
                        "count": 0,
                        "error": str(e)
                    }
            return info
        except Exception as e:
            print(f"❌ Failed to get collection information: {str(e)}")
            return {}
    
    def test_connection(self):
        """Test if the knowledge base connection is working properly"""
        try:
            # Attempt to list collections
            collections = self.client.list_collections()
            print(f"✅ Knowledge base connection is normal, with {len(collections)} collections in total")
            
            # Display collection information
            for collection in collections:
                try:
                    count = collection.count()
                    print(f"  📚 {collection.name}: {count} records")
                except Exception as e:
                    print(f"  ⚠️ {collection.name}: Unable to retrieve record count ({str(e)})")
            
            return True
        except Exception as e:
            print(f"❌ Knowledge base connection test failed: {str(e)}")
            return False
