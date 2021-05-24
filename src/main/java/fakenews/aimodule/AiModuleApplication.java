package fakenews.aimodule;

import fakenews.aimodule.entities.AiEntity;
import fakenews.aimodule.servicies.AiService;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class AiModuleApplication {

	public static void main(String[] args) {
		// SpringApplication.run(AiModuleApplication.class, args);
		AiService aiService = new AiService(null);
		try {
			aiService.calculateScore(new AiEntity(100, "lol", "lol=be"));
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
