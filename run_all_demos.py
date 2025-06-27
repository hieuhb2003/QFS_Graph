"""
Run All Demos - GraphRAG System
Script để chạy tất cả các demo của GraphRAG system
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from examples.demo_quick_test import quick_test
from examples.demo_full_workflow import demo_full_workflow, demo_simple_workflow
from examples.demo_clustering_integration import demo_clustering_integration
from examples.demo_graphrag import demo_graphrag
from src.graphrag.utils.logger_config import setup_logger


def print_menu():
    """In menu lựa chọn"""
    print("\n" + "="*60)
    print("🎯 GraphRAG System - Demo Menu")
    print("="*60)
    print("1. Quick Test (Test nhanh tất cả chức năng)")
    print("2. Full Workflow với LLM (Cần OpenAI API key)")
    print("3. Simple Workflow (Không cần LLM)")
    print("4. Clustering Integration Demo")
    print("5. GraphRAG Basic Demo")
    print("6. Chạy tất cả demo (trừ full workflow)")
    print("0. Thoát")
    print("="*60)


async def run_all_demos():
    """Chạy tất cả demo (trừ full workflow)"""
    logger = setup_logger(name="All-Demos", log_level="INFO")
    logger.info("🚀 Chạy tất cả demo...")
    
    demos = [
        ("Quick Test", quick_test),
        ("Simple Workflow", demo_simple_workflow),
        ("Clustering Integration", demo_clustering_integration),
        ("GraphRAG Basic", demo_graphrag)
    ]
    
    for name, demo_func in demos:
        try:
            logger.info(f"\n{'='*40}")
            logger.info(f"🎯 Running {name}...")
            logger.info(f"{'='*40}")
            await demo_func()
            logger.info(f"✅ {name} completed successfully!")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n🎉 All demos completed!")


async def main():
    """Main function"""
    logger = setup_logger(name="Demo-Runner", log_level="INFO")
    
    while True:
        print_menu()
        choice = input("Chọn demo (0-6): ").strip()
        
        try:
            if choice == "0":
                logger.info("👋 Goodbye!")
                break
            elif choice == "1":
                logger.info("🎯 Running Quick Test...")
                await quick_test()
            elif choice == "2":
                if not os.getenv("OPENAI_API_KEY"):
                    logger.warning("⚠️ Cần OpenAI API key để chạy full workflow!")
                    logger.info("💡 Set OPENAI_API_KEY environment variable")
                    continue
                logger.info("🎯 Running Full Workflow...")
                await demo_full_workflow()
            elif choice == "3":
                logger.info("🎯 Running Simple Workflow...")
                await demo_simple_workflow()
            elif choice == "4":
                logger.info("🎯 Running Clustering Integration...")
                await demo_clustering_integration()
            elif choice == "5":
                logger.info("🎯 Running GraphRAG Basic...")
                await demo_graphrag()
            elif choice == "6":
                await run_all_demos()
            else:
                logger.warning("❌ Lựa chọn không hợp lệ!")
                continue
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("🎯 GraphRAG System Demo Runner")
    print("="*50)
    print("📋 Available demos:")
    print("  • Quick Test: Test nhanh tất cả chức năng")
    print("  • Full Workflow: Demo đầy đủ với LLM")
    print("  • Simple Workflow: Demo không cần LLM")
    print("  • Clustering Integration: Demo clustering")
    print("  • GraphRAG Basic: Demo cơ bản")
    print("="*50)
    
    asyncio.run(main()) 