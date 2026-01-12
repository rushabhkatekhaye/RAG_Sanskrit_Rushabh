import requests
import json

class OllamaClient:
    def __init__(self, model="phi", base_url="http://localhost:11434"):
        """Initialize Ollama client
        
        Args:
            model: Model name (phi, gemma:2b, etc.)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def is_available(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(self.base_url, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt, max_tokens=300, temperature=0.7):
        """Generate response using Ollama
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return None
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    
    def test_connection(self):
        """Test Ollama connection"""
        print(f"Testing Ollama connection at {self.base_url}...")
        
        if not self.is_available():
            print("❌ Ollama is not running!")
            print("   Please start Ollama and try again.")
            return False
        
        print("✅ Ollama is running!")
        
        # Test generation
        print(f"Testing model: {self.model}...")
        response = self.generate("Hello", max_tokens=10)
        
        if response:
            print(f"✅ Model {self.model} is working!")
            return True
        else:
            print(f"❌ Model {self.model} not found!")
            print(f"   Run: ollama pull {self.model}")
            return False