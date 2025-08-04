import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class PodcastGenerator:
    def __init__(self, topic):
        self.topic = topic
        self.episode_number = 1
        self.host_name = "Overcomer"
        self.co_host_name = "Mighty Man"
        self.episode_title = f"Episode {self.episode_number}: {self.topic}"
        self.script = ""
        
        # Initialize the model and tokenizer
        print("Loading model...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate_script(self):
        """Generate a podcast script using TinyLlama"""
        try:
            prompt = f"""[INST] <<SYS>>
            You are a professional podcast script writer. Create an engaging podcast script about {self.topic}.
            
            Format:
            1. Introduction by {self.host_name}
            2. Discussion between {self.host_name} and {self.co_host_name}
            3. Key points about {self.topic}
            4. Conclusion
            
            Keep it conversational, informative, and about 500 words.
            Include some interesting facts or recent developments about {self.topic}.
            <</SYS>>[/INST]"""
            
            print("Generating script...")
            
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate text
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and clean up the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.script = generated_text.split('[/INST]')[-1].strip()
            return self.script
            
        except Exception as e:
            print(f"Error generating script: {e}")
            return None
    
    def save_script(self):
        """Save the generated script to a text file"""
        if not os.path.exists('episodes'):
            os.makedirs('episodes')
            
        filename = f"episodes/{self.topic.lower().replace(' ', '_')}_script.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Episode: {self.episode_title}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("-" * 50 + "\n\n")
            f.write(self.script)
        return filename
    
    def text_to_speech(self):
        """Convert the script to speech using gTTS"""
        if not self.script:
            print("No script available to convert to speech.")
            return None
            
        try:
            # Clean the script for TTS
            clean_script = self.script.replace('\n', ' ').replace('  ', ' ')
            
            # Generate speech
            tts = gTTS(text=clean_script, lang='en', slow=False)
            
            # Save the audio file
            if not os.path.exists('episodes'):
                os.makedirs('episodes')
                
            audio_file = f"episodes/{self.topic.lower().replace(' ', '_')}_podcast.mp3"
            tts.save(audio_file)
            return audio_file
            
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return None

def main():
    print("🎙️  Welcome to the AI Podcast Generator with TinyLlama!")
    print("This will use a local language model to generate your podcast.\n")
    
    topic = input("Enter the topic for your podcast (or press Enter for default): ").strip()
    
    if not topic:
        topic = "The Future of Artificial Intelligence"
        print(f"Using default topic: {topic}")
    
    # Create podcast generator instance
    print(f"\n🎤 Generating podcast about: {topic}")
    podcast = PodcastGenerator(topic)
    
    # Generate script
    print("📝 Creating script (this may take a few minutes)...")
    script = podcast.generate_script()
    
    if script:
        # Save script
        script_file = podcast.save_script()
        print(f"💾 Script saved to: {script_file}")
        
        # Convert to speech
        print("🔊 Converting to speech... (This may take a moment)")
        audio_file = podcast.text_to_speech()
        
        if audio_file:
            print(f"\n✅ Podcast generated successfully!")
            print(f"📄 Script: {script_file}")
            print(f"🔊 Audio: {audio_file}")
            print("\nTo play the podcast, open the audio file with any media player.")
        else:
            print("❌ Failed to generate audio. Please check the script file.")
    else:
        print("❌ Failed to generate podcast script. Please try again.")

if __name__ == "__main__":
    main()
