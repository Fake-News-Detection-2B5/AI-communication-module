package fakenews.aimodule.servicies;

import fakenews.aimodule.entities.AiEntity;
import fakenews.aimodule.entities.AiResultEntity;
import fakenews.aimodule.repositories.AiRepository;
import fakenews.aimodule.utilities.ScoreResult;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.Optional;
import java.util.Scanner;

@Service
public class AiService {

    private final AiRepository aiRepository;

    @Autowired
    public AiService(AiRepository aiRepository) {
        this.aiRepository = aiRepository;
    }

    /**
     * private method that returns the score of the aiEntity given
     * @param aiEntity the given entity for which the score needs to be calculated
     * @return the score of the parameter aiEntity
     * if the aiEntity given has already been checked, then we just return its score (which was calculated at some point before)
     * if it has not been checked before, then:
     * we execute calculate its score by executing the 2 python scripts with it as a parameter
     * we merge the 2 scores that result from the scripts
     * we insert the aiEntity into our detabase (id, title, content and score)
     * we return the resulted score (a ScoreResult object)
     */
    public ScoreResult getResult(AiEntity aiEntity) throws IOException {
        Optional<AiResultEntity> exists = aiRepository.findById(aiEntity.getId());

        if (exists.isEmpty()) {
            String result = calculateScore(aiEntity);

            AiResultEntity aiResult = new AiResultEntity();
            aiResult.setId(aiEntity.getId());
            aiResult.setTitle(aiEntity.getTitle());
            aiResult.setContent(aiEntity.getContent());
            aiResult.setResult(result);
            aiRepository.save(aiResult);

            return new ScoreResult(result);
        }
        else {
            return new ScoreResult(exists.get().getResult());
        }
    }
    /**
     * private method that gets the score resulted from the python script
     * @param aiEntity the given entity for which the score needs to be calculated
     * @return the score
     */
    public String calculateScore(AiEntity aiEntity) throws IOException {
        // Process p1 = Runtime.getRuntime().exec("python3 classification.py " + aiEntity.getContent());


        /*
        * For my fellow reader
        *
        * Dear friend,
        * the app works as expected. Maybe on the cloud too. I don't fucking know tho.
        * Good luck with it if you ever stumble upon this :)
        *
        * Best wishes,
        * myself from the past
        * */




        FileWriter input = new FileWriter("input.txt");

        input.write(aiEntity.getContent());
        input.flush();

        // Process p1 = Runtime.getRuntime().exec("python3 classification.py");
        Process p1 = Runtime.getRuntime().exec("python3 classification.py");
        try {
            p1.waitFor();//wait for py script to finish
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Scanner in = new Scanner(new File("scor.txt"));
        String score1 = in.nextLine();

        in.close();
        return score1;
    }
}
