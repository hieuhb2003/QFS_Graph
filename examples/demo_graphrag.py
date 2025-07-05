import asyncio
import os
from typing import Dict, Any

from src.graphrag.utils.logger_config import setup_logger, get_logger
from src.graphrag.core.graphrag_system import GraphRAGSystem
from src.graphrag.clients.llm_client import create_llm_client
from src.graphrag.clients.embedding_client import create_embedding_client

from dotenv import load_dotenv
load_dotenv()

async def demo_graphrag_system():
    """Demo hệ thống GraphRAG"""
    
    # Setup logger
    logger = setup_logger(
        name="GraphRAGDemo",
        log_level="INFO",
        log_dir="./logs"
    )
    
    # Cấu hình hệ thống
    working_dir = "./graphrag_data_2"
    global_config = {
        "working_dir": working_dir,
        "save_interval": 10,
        "max_retries": 3,
        "embedding_batch_num": 32,
        "embedding_dimension": 384,  # Dimension của all-MiniLM-L6-v2
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5
        }
    }
    
    # Tạo embedding client (có thể chọn một trong các options)
    # Option 1: Sentence Transformers (GPU acceleration)
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cpu"  # hoặc "cpu" nếu không có GPU
    )
    
    # Option 2: vLLM embedding (nếu có vLLM server)
    # embedding_client = create_embedding_client(
    #     client_type="vllm",
    #     model_name="llama2-7b-chat",
    #     url="http://localhost:8000/v1",
    #     api_key="dummy"
    # )
    
    # Option 3: OpenAI embedding
    # embedding_client = create_embedding_client(
    #     client_type="openai",
    #     model_name="text-embedding-ada-002",
    #     api_key="your-openai-api-key"
    # )
    
    # Tạo LLM client (có thể chọn một trong các options)
    # Option 1: vLLM client (self-hosted)
    # llm_client = create_llm_client(
    #     client_type="vllm",
    #     model_name="llama2-7b-chat",
    #     url="http://localhost:8000/v1",
    #     api_key="dummy"
    # )
    
    # Option 2: OpenAI client
    llm_client = create_llm_client(
        client_type="openai",
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Option 3: Không sử dụng LLM (chỉ dùng regex)
    # llm_client = None
    
    # Khởi tạo hệ thống GraphRAG
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=llm_client
    )
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "content": """
            Apple Inc. is a technology company that designs and manufactures consumer electronics. 
            The company was founded by Steve Jobs and Steve Wozniak in 1976. 
            Apple's most popular products include the iPhone, iPad, and Mac computers. 
            The iPhone is a line of smartphones designed by Apple Inc. and runs on iOS operating system.
            """
        },
        {
            "id": "doc2", 
            "content": """
            Microsoft Corporation is a multinational technology company. 
            Bill Gates and Paul Allen founded Microsoft in 1975. 
            Microsoft develops the Windows operating system and Office productivity software. 
            Windows is used by millions of people worldwide for personal and business computing.
            """
        },
        {
            "id": "doc3",
            "content": """
            Google LLC is a technology company that specializes in internet-related services. 
            Larry Page and Sergey Brin founded Google in 1998. 
            Google's main product is the Google Search engine, which helps users find information online. 
            The company also develops Android, a mobile operating system used by many smartphone manufacturers.
            """
        },
        {
            "id": "doc4",
            "content": """
            Tesla Inc. is an American electric vehicle and clean energy company. 
            Elon Musk is the CEO and co-founder of Tesla. 
            Tesla designs and manufactures electric cars, battery energy storage, and solar panels. 
            The company's most popular vehicle is the Tesla Model 3, which has become one of the best-selling electric cars worldwide.
            """
        },
        {
            "id": "doc5",
            "content": """
            Amazon.com Inc. is an American multinational technology company. 
            Jeff Bezos founded Amazon in 1994 as an online bookstore. 
            Amazon has grown to become the world's largest online retailer and cloud computing provider. 
            The company operates Amazon Web Services (AWS), which provides cloud computing services to businesses worldwide.
            """
        },
        {
            "id": "doc6",
            "content": """
            Facebook (now Meta Platforms Inc.) is a social media and technology company. 
            Mark Zuckerberg founded Facebook in 2004 while studying at Harvard University. 
            The company owns several popular social media platforms including Facebook, Instagram, and WhatsApp. 
            Meta is also developing virtual reality technology through its Oculus division.
            """
        },
        {
            "id": "doc7",
            "content": """
            Netflix Inc. is an American streaming service and production company. 
            Reed Hastings and Marc Randolph founded Netflix in 1997 as a DVD rental service. 
            Netflix has transformed into a leading streaming platform with millions of subscribers worldwide. 
            The company produces original content including popular series like Stranger Things and The Crown.
            """
        },
        {
            "id": "doc8",
            "content": """
            SpaceX is an American aerospace manufacturer and space transportation services company. 
            Elon Musk founded SpaceX in 2002 with the goal of reducing space transportation costs. 
            The company develops rockets and spacecraft, including the Falcon 9 rocket and Dragon spacecraft. 
            SpaceX has successfully launched numerous missions to the International Space Station.
            """
        },
        {
            "id": "doc9",
            "content": """
            NVIDIA Corporation is an American technology company that designs graphics processing units (GPUs). 
            Jensen Huang, Chris Malachowsky, and Curtis Priem founded NVIDIA in 1993. 
            NVIDIA's GPUs are widely used in gaming, artificial intelligence, and cryptocurrency mining. 
            The company's GeForce series of graphics cards is popular among gamers and content creators.
            """
        },
        {
            "id": "doc10",
            "content": """
            Intel Corporation is an American multinational technology company. 
            Gordon Moore and Robert Noyce founded Intel in 1968. 
            Intel is the world's largest semiconductor chip manufacturer by revenue. 
            The company produces processors for computers, servers, and other electronic devices, with its Core series being widely used in personal computers.
            """
        }
    ]

    docs = [
        # "San Antonio Spurs The San Antonio Spurs are an American professional basketball team based in San Antonio, Texas. The Spurs compete in the National Basketball Association (NBA) as a member of the league's Western Conference Southwest Division. The team plays its home games at the AT&T Center in San Antonio. The Spurs are one of four former American Basketball Association (ABA) teams to remain intact in the NBA after the 1976 ABA–NBA merger and the only former ABA team to have won an NBA championship. The Spurs' five NBA championships are the fifth most in history behind only the Boston",
        # "The arena seats 18,624 for basketball games and 17,565 for ice hockey games. The Bruins were the first American member of the National Hockey League and an Original Six franchise. The Boston Celtics were founding members of the Basketball Association of America, one of the two leagues that merged to form the NBA. The Celtics have the distinction of having won more championships than any other NBA team, with seventeen. While they have played in suburban Foxborough since 1971, the New England Patriots of the National Football League were founded in 1960 as the Boston Patriots, changing their name after",
        # "San Antonio Spurs là một đội bóng rổ chuyên nghiệp của Mỹ có trụ sở tại San Antonio, Texas. Spurs cạnh tranh trong Hiệp hội Bóng rổ Quốc gia (NBA) với tư cách là thành viên của Phân khu Tây Nam của Hội nghị Western. Đội chơi các trận đấu trên sân nhà tại Trung tâm AT&T ở San Antonio. Spurs là một trong bốn đội cũ của Hiệp hội Bóng rổ Mỹ (ABA) còn tồn tại trong NBA sau việc sáp nhập ABA-NBA năm 1976 và là đội cũ duy nhất của ABA đã giành chức vô địch NBA. Năm chức vô địch NBA của Spurs đứng thứ năm trong lịch sử, chỉ sau Boston",
        # "in the 16th century, Jesuits arrived in Beijing via Guangzhou. The most famous amongst them was Matteo Ricci, an Italian mathematician who came to China in 1588 and lived in Beijing. Ricci was welcomed at the imperial court and introduced Western learning into China. The Jesuits followed a policy of adaptation of Catholicism to traditional Chinese religious practices, especially ancestor worship. However, such practices were eventually condemned as polytheistic idolatry by the popes Clement XI, Clement XII and Benedict XIV. Roman Catholic missions struggled in obscurity for decades afterwards. Christianity began to take root in a significant way in the",
        # "or Mount Penglai. Han-era Daoists assembled into small groups of hermits who attempted to achieve immortality through breathing exercises, sexual techniques and use of medical elixirs. By the 2nd century AD, Daoists formed large hierarchical religious societies such as the Way of the Five Pecks of Rice. Its followers believed that the sage-philosopher Laozi (fl. 6th century BC) was a holy prophet who would offer salvation and good health if his devout followers would confess their sins, ban the worship of unclean gods who accepted meat sacrifices and chant sections of the \"Daodejing\". Buddhism first entered China during the Eastern",
        # "a major Buddhist centre by the middle of the 2nd century. Knowledge among people on the silk roads also increased when Emperor Ashoka of the Maurya dynasty (268–239 BCE) converted to Buddhism and raised the religion to official status in his northern Indian empire. From the 4th century CE onward, Chinese pilgrims also started to travel on the Silk Road to India to get improved access to the original Buddhist scriptures, with Fa-hsien's pilgrimage to India (395–414), and later Xuanzang (629–644) and Hyecho, who traveled from Korea to India. The travels of the priest Xuanzang were fictionalized in the 16th",
        "capital Luoyang in Emperor Ming's reign. Buddhism entered China via the Silk Road, transmitted by the Buddhist populations who inhabited the Western Regions (modern Xinjiang), then Indo-Europeans (predominantly Tocharians and Saka). It began to grow to become a significant influence in China proper only after the fall of the Han dynasty, in the period of political division. When Buddhism had become an established religion it began to compete with Chinese indigenous religion and Taoist movements, deprecatorily designated as Ways of Demons (鬼道 \"Guǐdào\") in Buddhist polemical literature. After the fall of the Han dynasty, a period of disunity defined as",
    
        # "San Antonio Spurs là một đội bóng rổ chuyên nghiệp của Mỹ có trụ sở tại San Antonio, Texas. Spurs cạnh tranh trong Hiệp hội Bóng rổ Quốc gia (NBA) với tư cách là thành viên của Phân khu Tây Nam của Hội nghị Western. Đội chơi các trận đấu trên sân nhà tại Trung tâm AT&T ở San Antonio. Spurs là một trong bốn đội cũ của Hiệp hội Bóng rổ Mỹ (ABA) còn tồn tại trong NBA sau việc sáp nhập ABA-NBA năm 1976 và là đội cũ duy nhất của ABA đã giành chức vô địch NBA. Năm chức vô địch NBA của Spurs đứng thứ năm trong lịch sử, chỉ sau Boston"
        # "Sân vận động có 18.624 chỗ ngồi cho các trận bóng rổ và 17.565 chỗ ngồi cho các trận khúc côn cầu trên băng. Bruins là thành viên Mỹ đầu tiên của National Hockey League và một trong sáu đội ban đầu. Boston Celtics là thành viên sáng lập của Hiệp hội Bóng rổ Mỹ, một trong hai giải đấu hợp nhất để hình thành NBA. Celtics có sự khác biệt khi đã giành được nhiều chức vô địch hơn bất kỳ đội nào khác của NBA, với mười bảy. Mặc dù họ đã chơi ở vùng ngoại ô Foxborough từ năm 1971, New England Patriots của Liên đoàn Bóng đá Quốc gia được thành lập vào năm 1960 với tên Boston Patriots, đổi tên sau",
        # "Vào thế kỷ 16, các nhà truyền giáo Dòng Tên đến Bắc Kinh qua Quảng Châu. Người nổi tiếng nhất trong số họ là Matteo Ricci, một nhà toán học người Ý đến Trung Quốc vào năm 1588 và sống ở Bắc Kinh. Ricci được chào đón tại triều đình hoàng gia và giới thiệu học thuật phương Tây vào Trung Quốc. Các nhà truyền giáo Dòng Tên theo đuổi một chính sách thích nghi của Công giáo với các phong tục tôn giáo truyền thống của Trung Quốc, đặc biệt là việc thờ cúng tổ tiên. Tuy nhiên, những phong tục như vậy cuối cùng đã bị các giáo hoàng Clement XI, Clement XII và Benedict XIV lên án là sự thờ lạy đa thần. Các sứ mệnh Công giáo Rôma đã đấu tranh trong bóng tối trong nhiều thập kỷ sau đó. Kitô giáo bắt đầu phát triển một cách đáng kể ở",
        # "hoặc Núi Penglai. Các nhà Đạo giáo thời Hán tụ họp thành các nhóm ẩn sĩ nhỏ cố gắng đạt được sự bất tử thông qua các bài tập thở, kỹ thuật tình dục và sử dụng thuốc tiên. Vào thế kỷ thứ 2 sau Công nguyên, các nhà Đạo giáo đã hình thành các xã hội tôn giáo phân cấp lớn như Con đường của Năm Pecks của Gạo. Những người theo đạo tin rằng nhà hiền triết Lão Tử (fl. Thế kỷ 6 trước Công nguyên) là một nhà tiên tri thiêng liêng sẽ mang lại sự cứu rỗi và sức khỏe tốt nếu những người theo đạo trung thành của ông thú nhận tội lỗi của mình, cấm việc thờ cúng các vị thần không sạch sẽ chấp nhận hy sinh thịt và đọc các phần của \"Đạo đức kinh\". Phật giáo lần đầu tiên进入 Trung Quốc trong thời kỳ Đông",
        # "một trung tâm Phật giáo lớn vào giữa thế kỷ thứ 2. Kiến thức giữa những người trên con đường tơ lụa cũng tăng lên khi Hoàng đế Ashoka của triều đại Maurya (268–239 TCN) chuyển sang Phật giáo và nâng tôn giáo lên vị thế chính thức trong đế chế Bắc Ấn của ông. Từ thế kỷ thứ 4 sau Công nguyên, các nhà sư Trung Quốc cũng bắt đầu du hành trên Con đường Tơ lụa đến Ấn Độ để có được quyền truy cập tốt hơn vào các kinh điển Phật giáo nguyên bản, với chuyến hành hương của Fa-hsien đến Ấn Độ (395–414), và sau đó là Xuanzang (629–644) và Hyecho, người đã du hành từ Hàn Quốc đến Ấn Độ. Những chuyến du hành của nhà sư Xuanzang đã được hư cấu trong thế kỷ 16",
        "thủ đô Lạc Dương trong triều đại của Hoàng đế Minh. Phật giáo du nhập vào Trung Quốc qua Con đường tơ lụa, được truyền bởi các dân tộc Phật tử cư trú tại các vùng phía Tây (Xinjiang hiện đại), sau đó là người Ấn-Âu (chủ yếu là người Tocharian và Saka). Nó bắt đầu phát triển để trở thành một ảnh hưởng đáng kể ở Trung Quốc chỉ sau khi nhà Hán sụp đổ, trong thời kỳ phân chia chính trị. Khi Phật giáo đã trở thành một tôn giáo được thành lập, nó bắt đầu cạnh tranh với các phong trào tôn giáo bản địa và Đạo giáo của Trung Quốc, được chỉ định một cách miệt thị là Con đường của Quỷ (鬼道 \"Guǐdào\") trong văn học tranh luận của Phật giáo."
        ]    

    try:
        # Insert documents
        logger.info("Starting batch document insertion...")
        
        # Chuẩn bị documents cho batch processing - chỉ cần list of contents
        # documents_batch = [doc['content'] for doc in documents]
        documents_batch = docs
        
        # Sử dụng batch processing thay vì tuần tự
        if llm_client:
            # Batch processing với LLM one-shot
            results = await system.insert_documents_batch_with_llm(
                documents=documents_batch,
                max_concurrent_docs=5  # Chạy song song tối đa 3 documents
            )
        else:
            # Batch processing với chunking
            results = await system.insert_documents_batch(
                documents=documents_batch,
                chunk_size=500,
                max_concurrent_docs=3  # Chạy song song tối đa 3 documents
            )
        
        # Kiểm tra kết quả
        success_count = sum(results)
        logger.info(f"Batch processing completed: {success_count}/{len(documents)} documents successful")
        
        for i, success in enumerate(results):
            if success:
                logger.info(f"Successfully processed document: {documents[i]['id']}")
            else:
                logger.error(f"Failed to process document: {documents[i]['id']}")
        
        # Query examples
        logger.info("\n" + "="*50)
        logger.info("QUERY EXAMPLES")
        logger.info("="*50)
        
        # Query entities
        logger.info("\n1. Querying entities...")
        entity_results = await system.query_entities("technology company", top_k=5)
        logger.info(f"Found {len(entity_results)} entities:")
        for i, result in enumerate(entity_results[:3]):
            logger.info(f"  {i+1}. {result.get('entity_name', 'N/A')}: {result.get('description', 'N/A')}")
        
        # Query relations
        logger.info("\n2. Querying relations...")
        relation_results = await system.query_relations("founded", top_k=5)
        logger.info(f"Found {len(relation_results)} relations:")
        for i, result in enumerate(relation_results[:3]):
            logger.info(f"  {i+1}. {result.get('source_entity', 'N/A')} -> {result.get('relation_description', 'N/A')} -> {result.get('target_entity', 'N/A')}")
        
        # Search entities by name
        logger.info("\n3. Searching entities by name...")
        apple_results = await system.search_entities_by_name("Apple")
        logger.info(f"Found {len(apple_results)} entities matching 'Apple':")
        for i, result in enumerate(apple_results[:2]):
            logger.info(f"  {i+1}. {result.get('entity_name', 'N/A')}: {result.get('description', 'N/A')}")
        
        # Get entity graph context
        logger.info("\n4. Getting entity graph context...")
        apple_context = await system.get_entity_graph_context("Apple")
        if "error" not in apple_context:
            logger.info(f"Apple entity context:")
            logger.info(f"  Description: {apple_context['entity']['data'].get('description', 'N/A')}")
            logger.info(f"  Neighbors: {apple_context['total_neighbors']}")
            for neighbor in apple_context['neighbors'][:3]:
                logger.info(f"    - {neighbor['name']}: {neighbor['relation']}")
        else:
            logger.warning(f"Could not get context for Apple: {apple_context['error']}")
        
        # Get system statistics
        logger.info("\n5. System statistics...")
        stats = await system.get_system_stats()
        logger.info(f"Document status: {stats.get('document_status', {})}")
        logger.info(f"Graph: {stats.get('graph', {})} nodes and edges")
        logger.info(f"Vector DBs: {stats.get('vector_dbs', {})} entities and relations")
        logger.info(f"Chunks: {stats.get('chunks', 0)}")
        
        # Get document status
        logger.info("\n6. Document status...")
        for doc in documents:
            status = await system.get_document_status(doc['id'])
            if status:
                logger.info(f"  {doc['id']}: {status.get('status', 'unknown')}")
                if status.get('status') == 'success':
                    logger.info(f"    Entities: {status.get('entities_count', 0)}, Relations: {status.get('relations_count', 0)}")
        
        logger.info("\n" + "="*50)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        await system.cleanup()
        logger.info("Demo completed!")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_graphrag_system()) 