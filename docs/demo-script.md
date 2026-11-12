# Demo script

Spoken lines for the recorded demo, split across three speakers. It follows
the sections of the [demo guide](demo-guide.md), and the files and requests
in [demo-csv](../demo-csv/README.md). Replace `[Speaker 1]`, `[Speaker 2]`
and `[Speaker 3]` with your names, and swap sections between speakers if you
like: the assignment only requires that every member speaks and appears on
camera.

Quoted text is spoken. *On screen* lines are actions, not speech. If the
screen shows a different number from the script, say the number on screen.
The speech alone takes about 8 minutes at a steady pace; clicks and waiting
for replies take the rest, so keep talking while the agent works.

| Part | Speaker | Target |
| --- | --- | --- |
| Introduction | 1 | 0:15 |
| 1. Model training and results | 1 | 2:00 |
| 2. System architecture | 1 | 0:45 |
| 3. Single-record prediction | 2 | 1:00 |
| 4. Batch prediction | 2 | 0:45 |
| 5. Labeled evaluation with conditional prediction | 2 | 2:00 |
| 6. Invalid input | 3 | 0:30 |
| 7. Agent evaluation | 3 | 0:45 |
| 8. Tests and deployment | 3 | 1:15 |
| Closing | 3 | 0:15 |

Total: about 9½ minutes.

Treat these times as targets. Rehearse with the live replies, uploads,
downloads and tab changes included, and trim pauses to stay within 8–10 minutes.

## Introduction · Speaker 1

*On screen: the live app's chat page.*

> "Hello, we are [Speaker 1], [Speaker 2] and [Speaker 3]. Our project
> classifies Windows PE files as malware or goodware with a machine-learning
> model we trained, and serves it through a chat agent that calls that model
> as a tool. Everything you will see runs on our live Render deployment."

## 1. Model training and results · Speaker 1

*On screen: Model dashboard, dataset and split section.*

> "We used the Brazilian malware dataset by Ceschin and colleagues: 50,181 PE
> files, each described by 27 fields and labeled malware or goodware.
>
> First, cleaning and leakage. Some files appear more than once, so we kept
> one row per SHA1 hash, and dropped 18 hashes whose copies had conflicting
> labels. That leaves 43,393 records. We also excluded the SHA1, the
> first-seen date, which reveals when each sample was collected, and three
> constant columns.
>
> Before any preprocessing, we reserved a stratified 20 percent hold-out:
> 8,679 records we did not touch until the end; the other 34,714 are for
> development. Imputation, a log transform, scaling, TF-IDF for the imported
> DLLs and symbols, and encoding of the packer and compiler field are all
> fitted inside each cross-validation fold through a scikit-learn pipeline, so
> validation data never leaks into them."

*On screen: step the cross-validation walkthrough from Round 1 to Round 2.*

> "We used stratified 10-fold cross-validation with seed 42. In each round,
> one fold validates and the other nine train, and every model sees the same
> folds.
>
> We trained seven models: logistic regression, a decision tree, a random
> forest, a PyTorch MLP, XGBoost, LightGBM and CatBoost."

*On screen: the seven-model comparison.*

> "Ranked by mean cross-validation AUC, computed from malware probabilities,
> the top three are LightGBM, with AUC 0.9982 and accuracy 98.57 percent;
> XGBoost, 0.9979 and 98.39 percent; and the random forest, 0.9975 and 98.24
> percent.
>
> The bottom two are logistic regression, 0.9886 and 95.77 percent, and the
> decision tree, 0.9802 and 97.57 percent. By accuracy alone, the tree would
> beat the MLP, but we rank by AUC, our primary metric."

*On screen: the selection panel, then the final-test panel.*

> "We chose LightGBM, froze it with a 0.5 threshold, and only then scored the
> hold-out: AUC 0.9979 and accuracy 98.43 percent."

## 2. System architecture · Speaker 1

*On screen: the Workflow tab.*

> "The browser uploads a CSV; our Flask server checks its type, size and columns, and keeps it in a private session.
> The raw rows never go to OpenAI.
>
> When we send a question, OpenAI's gpt-4.1-mini only chooses a tool,
> predict_single, predict_batch or evaluate, and its arguments. Python
> validates those arguments, runs our saved LightGBM pipeline and computes
> every metric, and the reply is built from the tool's output. So the language
> model never classifies anything itself, and with no explanation tool, it
> cannot invent explanations. Each file can hold up to
> 10,000 rows, and a session up to 5 files."

## 3. Single-record prediction · Speaker 2

*On screen: drag `01-single-record.csv` onto the upload area.*

> "I'm uploading a real CSV with one record. The app validates it and prepares
> a question for me, but nothing is sent until I click Send."

*On screen: change the question to `Classify row 0 of file <ID>.` and click
**Send**.*

> "The agent called predict_single, with this file's ID and row index 0. The
> model says goodware, with a malware probability of 0.0012, far below the
> 0.5 threshold. The activity record lists the call, its arguments and its
> result."

*On screen: send `What was the malware probability of that prediction?`*

> "For this follow-up, the agent answers from the stored result. There is no
> new tool call in the trail."

*On screen: send `Explain which individual features caused the prediction.`*

> "And if we ask why, the agent says no explanation tool exists, instead of
> guessing."

## 4. Batch prediction · Speaker 2

*On screen: click **02**, change the question to the batch request from the
demo README, and click **Send**.*

> "Now a batch of four records. The agent called predict_batch with the file
> ID: four rows classified, two malware and two goodware, none invalid."

*On screen: click **Download results** and open the file.*

> "The download has one row per record: its position, its SHA1, its status,
> the prediction and the malware probability."

*On screen: send `What is the malware probability of row 2?`, then
`What is the malware probability of row 1?`*

> "For each question, the agent called predict_single on that row of the same
> file. Row 2 is malware, with probability 0.97. Row 1 is goodware, with
> probability 0.0025."

## 5. Labeled evaluation with conditional prediction · Speaker 2

*On screen: click **Reset session & remove files**, then **04a** and **04b**.*

> "Now the conditional task. I've reset the session. 04a is a small labeled
> file; 04b is one record to predict."

*On screen: in the conditional form, choose 04a for evaluation, 04b for
prediction, row 0 and minimum evaluation accuracy 75%. Click **Prepare question**, then
**Send**.*

> "The server makes evaluate run first: AUC 1.0, accuracy 1.0, and the
> confusion matrix shows two goodware and two malware files, all correct. This
> tiny file shows how the app works; our model's quality comes from the
> hold-out. The AI then read that accuracy, saw that it meets 0.75, and called
> predict_single itself: goodware. The reply begins 'Condition met'."

*On screen: open **Workflow** in a new browser tab, at **Your last request**.*

> "In the Workflow tab, the second OpenAI call received the accuracy, 1.0,
> and chose predict_single. The server also checks that decision, so a
> prediction the threshold does not allow is blocked."

*On screen: back in the chat, click **05a** and **05b**. Prepare the same
question with 05a, 05b and minimum evaluation accuracy 90%, then click **Send**.*

> "Now the other outcome. 05a holds one record whose label we deliberately
> flipped, so the model gets it wrong. It is a control-flow test, not a
> quality measure. Evaluate runs first: accuracy 0, and AUC is unavailable
> because there is only one class. 0 is below 0.9, so predict_single is
> skipped, and the reply begins 'Prediction withheld'."

*On screen: send `How many false negatives were there in that evaluation?`*

> "One false negative: one malware file predicted as goodware. That answer
> came from the stored evaluation, with no new tool call."

## 6. Invalid input · Speaker 3

*On screen: click **Reset session & remove files**, then **07**, change the
question to the invalid-input request from the demo README, and click
**Send**.*

> "Now an error case. In this file, row 3 has the text 'invalid' in the
> numeric Size column. The agent called predict_batch: three rows classified,
> one invalid row, reported with its reason."

*On screen: download the results and open them.*

> "Row 3 is still in the download, in its original position, marked invalid
> with its error. Invalid rows are reported, never silently dropped."

## 7. Agent evaluation · Speaker 3

*On screen: agent-evaluation.md on GitHub.*

> "We also tested the agent with the real model. A runner sends 15 scenarios
> to the live site, covering every task, follow-ups and error cases, and
> checks each tool call and number against our saved model. Our latest run, Run 11 with gpt-4.1-mini, passed 15 of 15, and
> all 27 numeric checks matched. In an earlier set of 22 repeated conditional requests, the AI
> decided correctly every time.
>
> Earlier runs failed up to 2 of 15 scenarios, for example by running a tool
> again for a follow-up. Each failure led to a fix, and the next run passed
> all 15. The limitation is that the AI's choices vary between runs, which is
> why the server double-checks every conditional prediction."

## 8. Tests and deployment · Speaker 3

*On screen: a terminal running `docker compose run --rm ml pytest -q`, or the
test job log of a CI run.*

> "Our suite has 487 automated tests: unit tests for the training pipeline and
> tools, Flask integration tests, and agent tests with a mocked language
> model, including failures such as malformed CSVs, missing labels, a single
> class, and tool and provider errors."

*On screen: the GitHub Actions page and its latest run.*

> "On every push to main, GitHub Actions runs three test jobs: the full suite,
> the lean runtime suite and a production-container check. Only if all three
> pass does the deploy job trigger Render through a deploy hook. Render's own
> auto-deploy is off."

*On screen: [run 37962332078](https://github.com/mehdihamid1/ML-Project/actions/runs/37962332078).*

> "To prove the gate, we pushed a deliberately failing test. Both test jobs
> failed, the deploy job was skipped, and the live site stayed on the previous
> commit. Reverting it passed and deployed again, in run 37962725985."

*On screen: the latest run's deploy job log, then the live `/health` page.*

> "After deploying, the job polls the live health endpoint until it reports
> the tested commit. Here is that smoke test: status ok, our LightGBM model,
> OpenAI configured, and the same commit as the run."

## Closing · Speaker 3

> "To sum up: a LightGBM model with a hold-out AUC of 0.9979 and accuracy of
> 98.43 percent, served by an agent that only chooses tools, tested and
> deployed through a gated pipeline. The code and documentation are in our
> GitHub repository. Thank you."
