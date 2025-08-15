import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import jieba
from tqdm import tqdm

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DocumentProcessor")
jieba.initialize()

class DocumentProcessor:
    def __init__(self):
        # Model limitation parameters
        self.max_chars = 1500      # Character limit
        self.max_tokens = 450      # Token limit
        self.batch_size = 16       # Maximum number of texts per batch

    def _calculate_tokens(self, text: str) -> int:
        """Accurately calculate token count using jieba"""
        return len(list(jieba.cut(text)))

    def _split_sentences(self, text: str) -> List[str]:
        """Split into sentences while preserving semantic integrity"""
        sentences = re.split(r'(?<=[。！？；;])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _smart_chunking(self, text: str) -> List[str]:
        """Smart chunking strategy"""
        chunks = []
        current_chunk = []
        current_length = 0
        current_tokens = 0

        for sentence in self._split_sentences(text):
            sent_length = len(sentence)
            sent_tokens = self._calculate_tokens(sentence)

            # Double verification mechanism
            if (current_length + sent_length > self.max_chars) or \
               (current_tokens + sent_tokens > self.max_tokens):
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_length = 0
                current_tokens = 0

            current_chunk.append(sentence)
            current_length += sent_length
            current_tokens += sent_tokens

            # Forced chunk protection
            if sent_length > self.max_chars * 0.8:
                chunks.append(sentence[:self.max_chars])
                current_chunk = []
                current_length = 0
                current_tokens = 0

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _validate_chunk(self, chunk: str) -> bool:
        """Double verify text chunk"""
        if len(chunk) > self.max_chars:
            logger.warning(f"Character count exceeds limit: {len(chunk)}/{self.max_chars}")
            return False
        if self._calculate_tokens(chunk) > self.max_tokens:
            logger.warning(f"Token count exceeds limit: {self._calculate_tokens(chunk)}/{self.max_tokens}")
            return False
        return True

    def process_document(self, file_path: Path, output_dir: Path, 
                         custom_metadata: Optional[Dict[str, Any]] = None):
        """Process a single document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Smart chunking
            chunks = self._smart_chunking(content)
            valid_chunks = [c for c in chunks if self._validate_chunk(c)]

            # Construct output data
            output_data = []
            for idx, chunk in enumerate(valid_chunks):
                metadata = {
                    "source": str(file_path),
                    "chunk_id": idx,
                    "char_count": len(chunk),
                    "token_count": self._calculate_tokens(chunk)
                }
                
                # Add custom metadata
                if custom_metadata:
                    metadata.update(custom_metadata)

                output_data.append({
                    "text": chunk,
                    "metadata": metadata
                })

            # Save processing results
            output_path = output_dir / "processed_data.jsonl"
            with open(output_path, 'a', encoding='utf-8') as out_f:
                for item in output_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')

            logger.info(f"Successfully processed {file_path.name} => {output_path}")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")

    def batch_process(self, input_dir: str, output_dir: str, 
                      custom_metadata: Optional[Dict[str, Any]] = None):
        """Batch process documents"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob("*.txt"))
        if not files:
            logger.warning("No TXT files found")
            return

        logger.info(f"Starting processing {len(files)} documents...")
        for file in tqdm(files, desc="Processing"):
            self.process_document(file, output_path, custom_metadata)

if __name__ == "__main__":
    processor = DocumentProcessor()
    
    processor.batch_process(
        input_dir="./knowledge_data",
        output_dir="./processed_data"
    )
