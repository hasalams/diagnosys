# Example usage
import os
import asyncio
from src.multi_agent_pipeline import ContentIntelligencePipeline
from dotenv import load_dotenv
load_dotenv()

async def main():
    """Example usage of the Content Intelligence Pipeline"""
    
    # Check for required environment variable
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Please set this environment variable.")
        print("Example: export GOOGLE_API_KEY='your-api-key-here'")
        # continue anyway for demo; LLM calls will be attempted but may fail
        # return
    
    # Initialize pipeline
    pipeline = ContentIntelligencePipeline()
    
    # Example request
    request = "Create a comprehensive clinical evidence synthesis for the effectiveness of telemedicine interventions in managing Type 2 diabetes"
    
    print(f"Processing request: {request}\n")
    print("=" * 80)
    
    # Process the request
    result = await pipeline.process_request(request)
    
    # Display results
    if result["success"]:
        print("✅ Pipeline completed successfully!")
        print("\n📋 Final Output:")
        print("-" * 40)
        print(result["final_output"])
        
        print("\n📊 Quality Feedback:")
        if result["quality_feedback"]:
            for feedback in result["quality_feedback"].get("feedback", []):
                print(f"  • {feedback}")
        
        print(f"\n🔍 Agent History:")
        for activity in result["metadata"]["agent_history"]:
            print(f"  {activity['timestamp'][:19]} - {activity['agent']}: {activity['action']}")
            
    else:
        print("❌ Pipeline failed!")
        print(f"Error: {result.get('error')}")
        if result.get('errors'):
            for error in result['errors']:
                print(f"  • {error}")

if __name__ == "__main__":
    asyncio.run(main())