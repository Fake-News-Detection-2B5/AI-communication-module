package fakenews.aimodule;

import fakenews.aimodule.entities.AiEntity;
import fakenews.aimodule.servicies.AiService;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class AiModuleApplication {

	public static void main(String[] args) {
		SpringApplication.run(AiModuleApplication.class, args);
//		AiService aiService = new AiService(null);
//		try {
//			aiService.calculateScore(new AiEntity(1, "title of description", "title of lapis laluzi loool olololol"));
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
	}

}
