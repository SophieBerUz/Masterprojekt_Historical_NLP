# Introduction
Language is not static; it evolves and adapts, reflecting the cultural and historical contexts of its speakers. In the realm of computational linguistics, new dimensions are reached, as we explore the capabilities of modern language models to comprehend and interpret historical texts. For instance, previous research by González-Gallardo et al. (2023) explored the limitations of ChatGPT in identifying entities in historical documents, highlighting the difficulties language models face when processing texts from earlier times. <br>
We will assess how well generative language models perform when tasked with comprehending texts from different historical periods. To uncover this, we analyze how
accurately these models can predict linguistic features, such as part-of-speech (POS) tags, in historical German texts. Through this analysis, we aim to reveal the extent to which modern models can generalize across different time periods.

## Research Question

How well do language models generalize to historical German documents?

## Related Work

In recent years, significant advancements have been made in the application of Large Language Models (LLMs) to various natural language processing (NLP) tasks, particularly in sequence labeling and linguistic structure analysis across diverse languages.

One notable technique, TOPRO (Ma et al., 2024), is a sequence labeling technique that uses token-level prompts to capture token-specific features, boosting cross-lingual accuracy. This approach improves tasks like NER and POS tagging, particularly in languages with syntactic differences from English, by enhancing the model's adaptability to diverse linguistic contexts.
The paper Adelani et al. (2024) evaluated LLM performance in POS tagging for low-resource languages (LRLs), highlighting challenges due to limited data. Testing on twelve Brazilian LRLs and two African LRLs, the study finds that current LLMs struggle to achieve high accuracy without additional support, especially compared to high-resource languages like English and Portuguese.
Additionally, the study by Stüssi and Ströbel (20p24) evaluates GPT's effectiveness in tagging historical Latin texts, showing moderate success. The authors suggest that enhanced training with historical data and tailored adjustments could improve accuracy for both historical and low-resource languages.
The paper presented by Wang et al. (2024) examines LLMs in stemming tasks, evaluating Vocabulary, Contextual, and Entity-Based Contextual Stemming methods. While entity-based stemming outperforms traditional approaches, it is computationally costly, suggesting that optimized or hybrid methods are needed for effective LLM-based stemming.

Research such as Blevins et al. (2023) focuses on the ability of LLMs like GPT-3 to infer complex linguistic structures, including syntax and semantics, through carefully crafted prompts. This study highlights the potential of LLMs to uncover linguistic knowledge typically managed by rule-based systems. However, it primarily addresses English sequence tagging tasks, leaving an open question regarding the generalizability of these findings to other languages.
Additionally, the CARP model introduced by Sun et al. (2023) presents a systematic three-step approach to enhance text classification capabilities in LLMs. By decomposing the reasoning process, CARP allows models to better interpret nuanced text data, thereby facilitating accurate classifications even in complex linguistic scenarios.

Recent research shows that Large Language Models (LLMs) have potential in sequence labeling and text classification across languages, with methods like TOPRO and CARP improving accuracy by focusing on token-level nuances and structured classification. However, studies on low-resource languages and historical texts, such as Latin, reveal that LLMs often struggle without specialized training data, especially in contexts with limited resources or unique linguistic features. Building on these insights, our project examines LLM performance in POS tagging for German texts across different historical periods, aiming to understand how well LLMs can handle temporal language variation and contribute to historical linguistics.


# Methodology

Based on the research question, we came up with the following hypothesis:
"The accuracy of the predicted pos tags will decrease the further back we go in time." 
For the analysis, we utilized the DTA poem dataset, which consists of 22,039 German poetries and is annotated for Part-Of-Speech (POS). The period covered extends from the 17th to the 19th century. The time span was divided into 6 buckets each 50 years. 

The initial phase of this project involved reviewing two research papers [1,7], which provided three foundational prompt templates (template 1 and 2 from [1] and template 3 from [7]). These templates were translated into German and applied in initial tests on the Leo-Mistral[6] model with 10 samples. Based on initial results, additional prompt variations were introduced, including:
* template 4: like template 3 but add that we do not want extra information
* template 5: output should be only a list of the pos-tags
* template 6: like template 4 but change negativity to positivity “want only a list of tuples and nothing else”
* template 7: english version of template 3
* template 8: like template 3 but contains also a list and explanation of used tagset[4]
* template 9: combination of template 6 and 8
* template 10: like template 9 but with few-shot examples 

To enhance analysis, regex expressions were implemented to parse outputs and calculate accuracy, providing a systematic evaluation metric. Tests were then conducted on the Aya[2] model, albeit with only five samples due to runtime constraints. After assessing accuracy, the three most effective prompt templates were selected for further testing (results are in file Prompt_templates_results). <br>
Three best templates: <br>
```python
Template3: """Sie sind ein deutscher Linguist und Experte für Part-of-Speech Tagging. Sie verwenden STTS (Stuttgart-Tübingen Tagset). 
STTS Tags sind ADJA, ADJD, ADV, APPR, APPRART, APPO, APZR, ART, CARD, FM, ITJ, KOUI, KOUS, KON, KOKOM, NN, NE, PDS, PDAT, PIS, PIAT, PIDAT, PPER,
PPOS, PPOSAT, PRELS, PRELAT, PRF, PWS, PWAT, PWAV, PROAV, PTKZU, PTKNEG, PTKVZ, PTKANT, PTKA, VVFIN, VVIMP, VVINF, VVIZU, VVPP, VAFIN, VAIMP, VAINF,
VAPP, VMFIN, VMINF, VMPP, XY, $, , $(, $. und X. 
X steht für 'andere'.

Bitte geben Sie für jedes Wort im Eingabesatz die POS-Tags an. 
Die Eingabe besteht aus einer Liste von Wörtern im Satz. 
Das Ausgabeformat sollte eine Liste von Tupeln sein, wobei jedes Tupel aus einem Wort aus dem Eingabetext und der dazugehörigen POS-Markierung besteht. 
Ihre Antwort sollte nur eine Liste von Tupeln enthalten, in der Reihenfolge, in der die Wörter im Eingabesatz erscheinen, 
wobei jedes Tupel die entsprechende POS-Markierung für ein Wort enthält. 
Eingabesatz: {tokens}
"""
Template9: """Sie sind ein deutscher Linguist und Experte für Part-of-Speech Tagging. Sie verwenden STTS (Stuttgart-Tübingen Tagset). 
STTS Tags sind:
ADJA	attributives Adjektiv	[das] große [Haus]
ADJD	adverbiales oder prädikatives Adjektiv	[er fährt] schnell, [er ist] schnell 
ADV	Adverb	schon, bald, doch
APPR	Präposition; Zirkumposition links	in [der Stadt], ohne [mich]
APPRART	Präposition mit Artikel	im [Haus], zur [Sache]
APPO	Postposition	[ihm] zufolge, [der Sache] wegen
APZR	Zirkumposition rechts	[von jetzt] an
ART	bestimmter oder unbestimmter Artikel	der, die, das, ein, eine
CARD	Kardinalzahl	zwei [Männer], [im Jahre] 1994 
FM	Fremdsprachliches Material	[Er hat das mit ``] A big fish ['' übersetzt] 
ITJ	Interjektion	mhm, ach, tja
KOUI	unterordnende Konjunktion mit ``zu'' und Infinitiv	um [zu leben], anstatt [zu fragen]
KOUS	unterordnende Konjunktion mit Satz	weil, dass, damit, wenn, ob
KON	nebenordnende Konjunktion	und, oder, aber
KOKOM	Vergleichskonjunktion	als, wie
NN	normales Nomen	Tisch, Herr, [das] Reisen
NE	Eigennamen	Hans, Hamburg, HSV
PDS	substituierendes Demonstrativpronomen	dieser, jener
PDAT	attribuierendes Demonstrativpronomen	jener [Mensch]
PIS	substituierendes Indefinitpronomen	keiner, viele, man, niemand
PIAT	attribuierendes Indefinitpronomen ohne Determiner	kein [Mensch], irgendein [Glas]
PIDAT	attribuierendes Indefinitpronomen mit Determiner	[ein] wenig [Wasser], [die] beiden [Brüder] 
PPER	irreflexives Personalpronomen	ich, er, ihm, mich, dir
PPOSS	substituierendes Possessivpronomen	meins, deiner
PPOSAT	attribuierendes Possessivpronomen	mein [Buch], deine [Mutter]
PRELS	substituierendes Relativpronomen	[der Hund ,] der
PRELAT	attribuierendes Relativpronomen	[der Mann ,] dessen [Hund]
PRF	reflexives Personalpronomen	sich, einander, dich, mir
PWS	substituierendes Interrogativpronomen	wer, was
PWAT	attribuierendes Interrogativpronomen	welche[Farbe], wessen [Hut]
PWAV	adverbiales Interrogativ- oder Relativpronomen	warum, wo, wann, worüber, wobei
PAV	Pronominaladverb	dafür, dabei, deswegen, trotzdem
PTKZU	``zu'' vor Infinitiv	zu [gehen]
PTKNEG	Negationspartikel	nicht
PTKVZ	abgetrennter Verbzusatz	[er kommt] an, [er fährt] rad
PTKANT	Antwortpartikel	ja, nein, danke, bitte
PTKA	Partikel bei Adjektiv oder Adverb	am [schönsten], zu [schnell] 
TRUNC	Kompositions-Erstglied	An- [und Abreise]
VVFIN	finites Verb, voll	[du] gehst, [wir] kommen [an]
VVIMP	Imperativ, voll	komm [!]
VVINF	Infinitiv, voll	gehen, ankommen
VVIZU	Infinitiv mit ``zu'', voll	anzukommen, loszulassen
VVPP	Partizip Perfekt, voll	gegangen, angekommen
VAFIN	finites Verb, aux	[du] bist, [wir] werden
VAIMP	Imperativ, aux	sei [ruhig !]
VAINF	Infinitiv, aux	werden, sein
VAPP	Partizip Perfekt, aux	gewesen
VMFIN	finites Verb, modal	dürfen
VMINF	Infinitiv, modal	wollen
VMPP	Partizip Perfekt, modal	gekonnt, [er hat gehen] können
XY	Nichtwort, Sonderzeichen enthaltend	3:7, H2O, D2XW3 
$,	Komma	,
$.	Satzbeendende Interpunktion	. ? ! ; :
$(	sonstige Satzzeichen; satzintern	- [,]()
X	steht für 'andere'

Bitte geben Sie für jedes Wort im Eingabesatz die POS-Tags an. 
Die Eingabe besteht aus einer Liste von Wörtern im Satz. 
Das Ausgabeformat sollte eine Liste von Tupeln sein, wobei jedes Tupel aus einem Wort aus dem Eingabetext und der dazugehörigen POS-Markierung besteht. 
Ihre Antwort sollte nur eine Liste von Tupeln enthalten, in der Reihenfolge, in der die Wörter im Eingabesatz erscheinen, 
wobei jedes Tupel die entsprechende POS-Markierung für ein Wort enthält. 
Bitte geben Sie nur die Liste und nichts anderes. 
Eingabesatz: {tokens}
"""
Template10: """Sie sind ein deutscher Linguist und Experte für Part-of-Speech Tagging. Sie verwenden STTS (Stuttgart-Tübingen Tagset). 
STTS Tags sind:
ADJA	attributives Adjektiv	[das] große [Haus]
ADJD	adverbiales oder prädikatives Adjektiv	[er fährt] schnell, [er ist] schnell 
ADV	Adverb	schon, bald, doch
APPR	Präposition; Zirkumposition links	in [der Stadt], ohne [mich]
APPRART	Präposition mit Artikel	im [Haus], zur [Sache]
APPO	Postposition	[ihm] zufolge, [der Sache] wegen
APZR	Zirkumposition rechts	[von jetzt] an
ART	bestimmter oder unbestimmter Artikel	der, die, das, ein, eine
CARD	Kardinalzahl	zwei [Männer], [im Jahre] 1994 
FM	Fremdsprachliches Material	[Er hat das mit ``] A big fish ['' übersetzt] 
ITJ	Interjektion	mhm, ach, tja
KOUI	unterordnende Konjunktion mit ``zu'' und Infinitiv	um [zu leben], anstatt [zu fragen]
KOUS	unterordnende Konjunktion mit Satz	weil, dass, damit, wenn, ob
KON	nebenordnende Konjunktion	und, oder, aber
KOKOM	Vergleichskonjunktion	als, wie
NN	normales Nomen	Tisch, Herr, [das] Reisen
NE	Eigennamen	Hans, Hamburg, HSV
PDS	substituierendes Demonstrativpronomen	dieser, jener
PDAT	attribuierendes Demonstrativpronomen	jener [Mensch]
PIS	substituierendes Indefinitpronomen	keiner, viele, man, niemand
PIAT	attribuierendes Indefinitpronomen ohne Determiner	kein [Mensch], irgendein [Glas]
PIDAT	attribuierendes Indefinitpronomen mit Determiner	[ein] wenig [Wasser], [die] beiden [Brüder] 
PPER	irreflexives Personalpronomen	ich, er, ihm, mich, dir
PPOSS	substituierendes Possessivpronomen	meins, deiner
PPOSAT	attribuierendes Possessivpronomen	mein [Buch], deine [Mutter]
PRELS	substituierendes Relativpronomen	[der Hund ,] der
PRELAT	attribuierendes Relativpronomen	[der Mann ,] dessen [Hund]
PRF	reflexives Personalpronomen	sich, einander, dich, mir
PWS	substituierendes Interrogativpronomen	wer, was
PWAT	attribuierendes Interrogativpronomen	welche[Farbe], wessen [Hut]
PWAV	adverbiales Interrogativ- oder Relativpronomen	warum, wo, wann, worüber, wobei
PAV	Pronominaladverb	dafür, dabei, deswegen, trotzdem
PTKZU	``zu'' vor Infinitiv	zu [gehen]
PTKNEG	Negationspartikel	nicht
PTKVZ	abgetrennter Verbzusatz	[er kommt] an, [er fährt] rad
PTKANT	Antwortpartikel	ja, nein, danke, bitte
PTKA	Partikel bei Adjektiv oder Adverb	am [schönsten], zu [schnell] 
TRUNC	Kompositions-Erstglied	An- [und Abreise]
VVFIN	finites Verb, voll	[du] gehst, [wir] kommen [an]
VVIMP	Imperativ, voll	komm [!]
VVINF	Infinitiv, voll	gehen, ankommen
VVIZU	Infinitiv mit ``zu'', voll	anzukommen, loszulassen
VVPP	Partizip Perfekt, voll	gegangen, angekommen
VAFIN	finites Verb, aux	[du] bist, [wir] werden
VAIMP	Imperativ, aux	sei [ruhig !]
VAINF	Infinitiv, aux	werden, sein
VAPP	Partizip Perfekt, aux	gewesen
VMFIN	finites Verb, modal	dürfen
VMINF	Infinitiv, modal	wollen
VMPP	Partizip Perfekt, modal	gekonnt, [er hat gehen] können
XY	Nichtwort, Sonderzeichen enthaltend	3:7, H2O, D2XW3 
$,	Komma	,
$.	Satzbeendende Interpunktion	. ? ! ; :
$(	sonstige Satzzeichen; satzintern	- [,]()
X	steht für 'andere'	

Bitte geben Sie für jedes Wort im Eingabesatz die POS-Tags an. 
Die Eingabe besteht aus einer Liste von Wörtern im Satz. 
Das Ausgabeformat sollte eine Liste von Tupeln sein, wobei jedes Tupel aus einem Wort aus dem Eingabetext und der dazugehörigen POS-Markierung besteht. 
Ihre Antwort sollte nur eine Liste von Tupeln enthalten, in der Reihenfolge, in der die Wörter im Eingabesatz erscheinen, 
wobei jedes Tupel die entsprechende POS-Markierung für ein Wort enthält. 
Bitte geben Sie nur die Liste und nichts anderes zurück. 
Hier sind ein paar Beispiele, was Sie machen sollen:
Eingabesatz: ['Balde', 'rings', 'die', 'Sterne', 'bleichen', 'Und', 'ermattet', 'von', 'Beschwerde', 'Wächsern', 'ihre', 'Wangen', 'bleichen', '.', 'Fäulnis', 'wittert', 'aus', 'der', 'Erde', '.']
Ausgabe: [('Balde','NN'),('rings','ADV'),('die','ART'),('Sterne','NN'),('bleichen','VVINF'),('Und','KON'),('ermattet','VVFIN'),('von','APPR'),('Beschwerde','NN'),('Wächsern','KOUS'),('ihre','PPOSAT'),('Wangen','NN'),('bleichen','VVINF'),('.','$.'),('Fäulnis','NN'),('wittert','VVFIN'),('aus','APPR'),('der','ART'),('Erde','NN'),('.','$.')]

Eingabesatz: ['Traurig', 'rauscht', 'das', 'Rohr', 'im', 'Tümpel', 'Und', 'sie', 'friert', 'in', 'sich', 'gekauert', '.', 'Fern', 'ein', 'Hahn', 'kräht', '.', 'Übern', 'Tümpel', 'Hart', 'und', 'grau', 'der', 'Morgen', 'schauert', '.']
Ausgabe: [('Traurig','ADJD'), ('rauscht','VVFIN'), ('das','ART'), ('Rohr','NN'), ('im','APPRART'), ('Tümpel','NN'), ('Und','KON'), ('sie','PPER'), ('friert','VVFIN'), ('in','APPR'), ('sich','PRF'), ('gekauert','VVPP'), ('.','$.'), ('Fern','NN'),('ein','ART'), ('Hahn','NN'), ('kräht','VVFIN'), ('.','$.'), ('Übern','NN'), ('Tümpel','NN'), ('Hart','NE'), ('und','KON'), ('grau','ADJD'), ('der','ART'), ('Morgen','NN'), ('schauert','VVFIN'), ('.','$.')]

Eingabesatz: ['In', 'der', 'Schmiede', 'dröhnt', 'der', 'Hammer', 'Und', 'sie', 'huscht', 'am', 'Tor', 'vorüber', '.', 'Glührot', 'schwingt', 'der', 'Knecht', 'den', 'Hammer', 'Und', 'sie', 'schaut', 'wie', 'tot', 'hinüber', '.']
Ausgabe: [('In','APPR'), ('der','ART'), ('Schmiede','NN'), ('dröhnt','VVFIN'), ('der','ART'), ('Hammer','NN'), ('Und','KON'), ('sie','PPER'), ('huscht','VVFIN'), ('am','APPART'), ('Tor','NN'), ('vorüber','PTKVZ'), ('.','$.'), ('Glührot','NN'), ('schwingt','VVFIN'), ('der','ART'), ('Knecht','NN'), ('den','ART'), ('Hammer','NN'), ('Und','KON'), ('sie','PPER'), ('schaut','VVFIN'), ('wie','KOKOM'), ('tot','ADJD'), ('hinüber','ADV'), ('.','$.')]
Eingabesatz: {tokens}"""
```
Subsequent experiments on the Leo-Mistral model utilized 20 and 100 samples per bucket, aiming to understand how accuracy trends evolve over time. Finally, experiments were also conducted with 500 samples per bucket on both models. During testing on Aya, the dataset was refined by removing samples exceeding a specified length (text length histogram, see in _histogram_text lengths.ipynb_ ).
After that different micro and macro analysis was done on the generated data.  The experimental data was visualized across four primary plots: 
1. accuracy per bucket for each model, 
2. accuracy per bucket with error bars,
3. a correlation plot with a slope indicating performance trends over the years, and 
4. accuracy per token per year,

all created for each of the three selected templates. 


# Implementation Details

## Files

**src/corpora:**
* _dta_corpus_texts.json_ - contains poems as full text
* _dta_corpus_tokens.json_ - contains poems tokenized
* _poems_complete_with_buckets.json_ - combines two datasets above and states the bucket for each poem

**preprocessing:**
* _poems_dataset_preprocessing.ipynb_ - loads text and corpus, process and stores _poems_complete_with_buckets

**pipelines:**
* _pipeline_small_dataset_leoMistral.ipynb_ - run templates on 10 samples, calculate average accuracy for each template
* _pipeline_small_dataset_aya23_8B.ipynb_ - run templates on 10 samples, calculate average accuracy for each template
* _leo-mistral_autorun.ipynb_ - sample data for 20/100/500 samples per bucket, run experiment, store results and calculate accuracy per bucket
* _aya23_8B-autorun.ipynb_ - run experiments for 20/100/500 samples per bucket , store results and calculate accuracy per bucket <br>
_leo-mistral_autorun.ipynb_ is used to sample the data for either 20, 100 or 500 samples per bucket. By commenting this code section out, the code can be run either on _20_samples_perBucket_data.json_, _100_samples_perBucket_data.json_ or _500_samples_DATA.json_. <br>
In _aya23_8B-autorun.ipynb_ you can either load _20_samples_perBucket_data.json_, _100_samples_perBucket_data.json_ or _500_samples_DATA to run experiment.

**datasets:**
* _20_samples_perBucket_data.json_ - dataset with 20 poems/samples per bucket
* _100_samples_perBucket_data.json_ - dataset with 100 poems/samples per bucket
* _500_samples_DATA.json_ - dataset with 500 poems/samples per bucket
* _500_samples_DATA_removedTextsLarger8319characters.json_ - dataset with 500 samples per bucket for aya model, without samples that larger than 8319

**results:**

* _Prompt_templates_results.docx_ - stores for both models, the accuracies of each poem from the 10 tested templates and the average accuracy per template 

leo-mistral:
* _20_samples_perBucket_RESULTS.json_ - contains generated output
* _20_samples_perBucket_RESULTS_extracted.json_ - contains additionally parsed output 
* _20_samples_RESULTS_extracted_accuracy.json_ - contains additionally accuarcy for each poem
* _accuracy_perBucket_perTemplate_20samples.json_ - contains accuarcy for each bucket
* _100_samples_perBucket_RESULTS.json_ - contains generated output
* _100_samples_perBucket_RESULTS_extracted.json_ - contains additionally parsed output
* _100_samples_RESULTS_extracted_accuracy.json_ - contains additionally accuarcy for each poem
* _accuracy_perBucket_perTemplate_100samples.json_ - contains accuracy for each bucket
* _500_samples_RESULTS.json_ - contains generated output
* _500_samples_RESULTS_extracted.json_ - contains additionally parsed output
* _500_samples_RESULTS_extracted_accuracy.json_ - contains additionally accuarcy for each poem
* _500_samples_ACCURACY.json_ - contains accuracy for each bucket
* _accuracy_per_token_leomistral.json_ - contains accuarcy for each year and template 

aya23_8B:
* _500_samples_RESULTS_aya23.json_ - contains generated output
* _500_samples_RESULTS_aya23_extracted.json_ - contains additionally parsed output
* _500_samples_RESULTS_extracted_accuracy_AYA.json_ - contains additionally accuarcy for each poem
* _500_samples_ACCURACY_aya23.json_ - contains accuarcy for each bucket
* _accuracy_per_token_ayamodel.json_ - contains accuarcy for each year and template


**visualizations:**
* _visualization.ipynb_ - creates the four visualizations
* _histogram_text lengths.ipynb_ - creates distribution of text lengths
* _analysis_peak_accuracy_per_token.ipynb_ - contains analysis for peak sighted in visualization accuarcy per token

**src/corpora**, **datasets** and **results** are stored in a zip file in the shared Google Drive folder

## Setup
For the project the python version 3.10.8 was used. <br>
```console
curl https://pyenv.run | bash
pyenv install 3.10.8
```

Create and start an enviroment with: <br> 
```console
pyenv virtualenv 3.10.8 venvname
pyenv activate venvname
```

## Run Experiments
If you want to rerun the experiments, execute the files in the following order and choose the associated datasets: <br>
for leo-mistral: <br>
* _leo-mistral_autorun.ipynb_ - add _poems_complete_with_buckets.json_ to your environment, choose size of bucket -> the output will be four files like _20_samples_perBucket_RESULTS.json_, _20_samples_perBucket_RESULTS_extracted.json_, _20_samples_RESULTS_extracted_accuracy.json_ and _accuracy_perBucket_perTemplate_20samples.json_  
* _visualization.ipynb_ - uses the two files like _20_samples_perBucket_RESULTS_extracted.json_ and _20_samples_RESULTS_extracted_accuracy.json_ -> output will give you the four plots explained in **Methodology** and also file like _accuracy_per_token_leomistral.json_ 

for aya23:<br>
* _aya23_8B-autorun.ipynb_ - add _poems_complete_with_buckets.json_ to your environment, choose size of bucket -> the output will be tour files like _500_samples_RESULTS_aya23.json_, _500_samples_RESULTS_aya23_extracted.json_, _500_samples_RESULTS_extracted_accuracy_AYA.json_,_500_samples_ACCURACY_aya23.json_  
* _visualization.ipynb_ - uses the two files like _500_samples_RESULTS_aya23_extracted.json_ and _500_samples_ACCURACY_aya23.json_ -> output will give you the four plots explained in **Methodology** and also file like _accuracy_per_token_ayamodel.json_

* _analysis_peak_accuracy_per_token.ipynb_ - will need _accuracy_per_token_ and _extracted_ dataset from both models -> will give you two plots with accuracy per token for each model

# Results
We started our analysis with an evaluation of the three best templates on the leo-mistral model using 20 samples per bucket and 100 samples per bucket. <br>
The following plot compares the performance of the three prompt templates over different time intervals (“buckets”). Template 10 outperforms the other two templates across most buckets, showing more dynamic changes in accuracy. Templates 3 and 9 maintain more consistent but lower accuracy levels, with both templates experiencing minor improvements over time. The notable dip around 1700-1749 could indicate a time-specific challenge for all templates.
![20samples leo-mistral](figures/leo-mistral/20samples_accuracy_combined.png)
*caption: Accuracy for 20 samples on leo-mistral model* <br>

Increasing the sample size from 20 to 100 per bucket
![100samples leo-mistral](figures/leo-mistral/100samples_accuracy_combined.png)
*caption: Accuracy for 100 samples on leo-mistral model* <br>
results in smoother accuracy trends, with reduced variability between buckets. This suggests that the larger sample size provides a more stable estimate of each template’s performance over time. Template 10 still achieves the highest accuracy overall, but with less dramatic fluctuations, while Templates 3 and 9 maintain steady, lower accuracy levels. This trend implies that Template 10’s effectiveness remains superior, but the increased sample size makes the performance differences across time periods more reliable and less susceptible to noise. <br>

The final evaluation for the two models was done with 500 samples per bucket. The following table will show some example outputs of both models: <br>
For leo-mistral, we can see even if the template is the same, the output looks very different. In the first row the model translated the pos-tags from the STTS tag system to the IF tag system. The third row output contains a combination of tokens that is not from the input.

| template | template_id leo | template_id aya
| ------ | ------ | ------ |
|   3     |     1   |	2	|
|    9    |      2  |	9	|
|   10	  |		  3 |	10	|
```json:table
{ "items": [
		{"year":1641,"bucket":"1600-1649","tokens":["Wer","ist","doch","jmmer","so","geschossen","\/","Da\u00df","ab","dem","lieben","rebensafft","(","der","vnsers","hertzens","trost","vnd","krafft",")","Er","vnw\u00fcrsch","sein","solt","vnd","verdrossen","?"],"labels":["PWS","VAFIN","ADV","ADV","ADV","VVPP","$(","KOUS","APPR","ART","ADJA","NN","$(","ART","ADJA","NN","VVFIN","KON","NN","$(","PPER","ADJD","VAINF","VMFIN","KON","VVINF","$."],"template_id":1,"POS_Result":"Wir \u00fcbersetzen das gegebene POS-Tag-System in das IF-Tag-System, das aus den Tags ADJ, N, V, ART, COU, KON, ITJ, PREP, POS, PRO, PN, PRP, PDT, PMP, PM, POJ, POT, PRV, PW, X, $ besteht. \nDas Format ist wie folgt: {Tag: Anzahl der Vorkommen des Tags im Satz}. \nDie \u00dcbersetzung der Tags ist wie folgt:\n\nADJA -> ADJ\nADJD -> ADJ\nADV -> ADV\nAPPR -> PRV\nAPPO -> POJ\nAPZR -> PRV\nART -> ART\nCARD -> X\nFM -> X\nITJ -> ITJ\nKOUI -> KON\nKOUS -> COU\nKON -> KON\nNNS -> N\nNE -> X\nPDS -> PDT\nPDAT -> PDT\nPIS -> X\nPIAT -> ITJ\nPIDAT -> X\nPPER -> PRO\nPPOS -> POS\nPPOSAT -> POS\nPRELS -> PREP\nPRELAT -> PREP\nPRF -> PRO\nPWS -> POJ\nPWAT -> X\nPWAV -> X\nPROAV -> PRO\nPTKZU -> PRO\nPTKNEG -> X\nPTKVZ -> PRO\nPTKANT -> PRO\nPTKA -> X\nVVFIN -> VV\nVVIMP -> VV\nVVINF -> VV\nVVIZU -> VV\nVVPP -> X\nVAFIN -> X\nVAIMP -> X\nVAINF -> X\nVAPP -> X\nVMFIN -> X\nVMINF -> X\nVMPP -> X\nXY -> X\n$ \n","extracted_pos_tags":[]},
{"year":1641,"bucket":"1600-1649","tokens":["Ihr","G\u00f6ttin","zart","\/","jhr","deren","herrlichkeit","Die","G\u00f6tter","selbs","nicht","k\u00f6nden","widerstreben","\/","Ach","stehet","ab","von","ewrer","h\u00e4rtigkeit","Vnd","lasset","doch","den","Rittern","hie","das","leben","."],"labels":["ADV","PPER","ADV","VVFIN","APPR","ART","NN"],"template_id":3,"POS_Result":"[('Dan','NN'),('er','PPER'),('kaum','ADV'),('r\u00fcnnet','VVFIN'),('au\u00df','APPR'),('dem','ART'),('loch','NN')] \n","extracted_pos_tags":["NN","PPER","ADV","VVFIN","APPR","ART","NN"]},
{"year":1641,"bucket":"1600-1649","tokens":["Was","wunder","ists","mein","Lieb","\/","da\u00df","wir","dir","bl\u00fchen","sehen","Die","Rosen","\/","da","wir","doch","im","kalten","Winter","sein","\/","Es","ist","genug","da\u00df","sie","dein","Athem","an","thut","wehen","\/","Vnd","deiner","Augen","Liecht","ist","jhnen","Sonnenschein","."],"labels":["PIS","VVFIN","ADV","ADJD","$(","PPER","VVFIN","APPR","ADJA","NN","KON","NN","$(","ADV","ADV","APPR","APPRART","NN","PPOSAT","NN","ART","NN","ART","APPR"],"template_id":1,"POS_Result":"Die Part-of-Speech Tags f\u00fcr die W\u00f6rter im Eingabesatz sind wie folgt: \n['Nichts', 'ligt','so', 'tieff', '\/', 'es','sucht', 'zu', 'letzte', 'Lufft', 'vnd', 'Licht', '\/', 'Da', 'al\u00df', 'vmb', 'Jm', 'Nebel','seinen', 'Leib', 'der', 'Stund', 'der', 'von'] \n\nDiese Tags verwenden das STTS-Tagset, das von dem deutschen Linguisten und Part-of-Speech-Experten entwickelt wurde. \nDie Tags helfen, die Rolle oder Funktion jedes Wortes im Satz zu identifizieren und zu klassifizieren. \n","extracted_pos_tags":[]}
	]
}

```
The outputs for the aya model seem more reasonable. In the first example the model tried to determine the pos-tags, but ended up in a list with a mixture of tokens and tags, where not every token has a tag.: <br>
```json:table
{"items": [
		{"year":1641,"bucket":"1600-1649","tokens":["Wer","ist","doch","jmmer","so","geschossen","\/","Da\u00df","ab","dem","lieben","rebensafft","(","der","vnsers","hertzens","trost","vnd","krafft",")","Er","vnw\u00fcrsch","sein","solt","vnd","verdrossen","?"],"labels":["PWS","VAFIN","ADV","ADV","ADV","VVPP","$(","KOUS","APPR","ART","ADJA","NN","$(","ART","ADJA","NN","VVFIN","KON","NN","$(","PPER","ADJD","VAINF","VMFIN","KON","VVINF","$."],"template_id":2,"POS_Result":"Hier ist die Liste der W\u00f6rter mit ihren entsprechenden POS-Tags: \n\n['Wer', ADJA, 'ist', VAFIN, 'doch', ADJA, 'jmmer', ADJA, 'so', ADV, 'geschossen', VVIMP, '\/', 'Da\u00df', ADJA, 'ab', 'dem', 'lieben', 'rebensafft', '(', 'der', ART, 'vnsers', 'hertzens', 'trost', 'vnd', 'krafft', ')', 'Er', ADJA, 'vnw\u00fcrsch', 'sein', VAFIN, 'solt', VAFIN, 'vnd', 'verdrossen', '?']","extracted_pos_tags":[]},
{"year":1641,"bucket":"1600-1649","tokens":["Ihr","G\u00f6ttin","zart","\/","jhr","deren","herrlichkeit","Die","G\u00f6tter","selbs","nicht","k\u00f6nden","widerstreben","\/","Ach","stehet","ab","von","ewrer","h\u00e4rtigkeit","Vnd","lasset","doch","den","Rittern","hie","das","leben","."],"labels":["ADV","PPER","ADV","VVFIN","APPR","ART","NN"],"template_id":10,"POS_Result":"[('Dan','NE'),('er','PERS'),('kaum','ADV'),('r\u00fcnnet','VVFIN'),('au\u00df','APZR'),('dem','APPR'),('loch','NN')]","extracted_pos_tags":["NE","PERS","ADV","VVFIN","APZR","APPR","NN"]},
{"year":1641,"bucket":"1600-1649","tokens":["Was","wunder","ists","mein","Lieb","\/","da\u00df","wir","dir","bl\u00fchen","sehen","Die","Rosen","\/","da","wir","doch","im","kalten","Winter","sein","\/","Es","ist","genug","da\u00df","sie","dein","Athem","an","thut","wehen","\/","Vnd","deiner","Augen","Liecht","ist","jhnen","Sonnenschein","."],"labels":["PIS","VVFIN","ADV","ADJD","$(","PPER","VVFIN","APPR","ADJA","NN","KON","NN","$(","ADV","ADV","APPR","APPRART","NN","PPOSAT","NN","ART","NN","ART","APPR"],"template_id":2,"POS_Result":"Hier ist die Liste der W\u00f6rter mit ihren entsprechenden POS-Tags:\n\n[('Nichts', 'NN'), ('ligt', 'VAFIN'), ('so', 'ADV'), ('tieff', 'ADV'), ('\/', 'PTKZU'), ('es', 'PROAV'), ('sucht', 'VVFIN'), ('zu', 'PRELS'), ('letzte', 'KOKOM'), ('Lufft', 'NN'), ('vnd', 'APPR'), ('Licht', 'NN'), ('\/', 'PTKZU'), ('Da', 'APPR'), ('al\u00df', 'ADV'), ('vmb', 'APPR'), ('Jm', 'NN'), ('Nebel', 'NN'), ('seinen', 'APPR'), ('Leib', 'NN'), ('der', 'ART'), ('Stund', 'NN'), ('der', 'ART'), ('von', 'APPR')]","extracted_pos_tags":["NN","VAFIN","ADV","ADV","PTKZU","PROAV","VVFIN","PRELS","KOKOM","NN","APPR","NN","PTKZU","APPR","ADV","APPR","NN","NN","APPR","NN","ART","NN","ART","APPR"]}

]}
```
To get a better idea of how well each model performed, we started with the accuracy per bucket for leo-mistral: <br>
![500samples leo-mistral](figures/leo-mistral/500samples_accuracy.png)
*caption: Accuracy for 500 samples on leo-mistral model* <br>
Template 10 consistently shows the highest accuracy across all buckets, starting around 17.5% in the 1600-1649 bucket and gradually increasing to about 27.5% in the 1850-1899 bucket. Template 9 maintains moderate accuracy, consistently outperforming Template 3 but staying below Template 10. The accuracy trend for Template 9 is relatively stable with a gradual increase over time. Template 3 starts with the lowest accuracy among the three templates at just over 7.5% in the 1600-1649 bucket. This visualization suggests that Template 10 has a marked advantage in terms of accuracy across historical time periods. <br>
Comparing with the aya8B model, we can see some differences: <br>
![500samples aya8B](figures/aya8B/500samples_accuracy_AYA.png)
*caption: Accuracy for 500 samples on aya8B model* <br>
The accuracy values in this graph are generally higher than in the previous visualization. For instance, Template 10 starts around 30% accuracy here (compared to 17.5% in the previous chart) and reaches around 45%, suggesting this model is more accurate overall. In the previous visualization, Template 10 showed a consistent increase, with a slight drop in the 1700-1749 bucket, then continued upward. Here, Template 10 rises more steadily, peaks in the 1800-1849 bucket, and shows a slight drop in the final bucket. Template 9 also shows a similar trend, with an initial rise followed by a slight decline in the later buckets, indicating possible model performance degradation over time.

In both visualizations, Template 9 performs better than Template 3 but remains below Template 10.
However, the gap between Template 3 and Template 9 appears somewhat larger here, indicating that Template 9 performs relatively better in the aya8B model. <br>

In a next step, we did some more detailed evaluations. One analysis contained the standard error for each bucket, which is relativly small for both models but a little bit bigger for aya8B: <br>
![500samples leo-mistral standardError](figures/leo-mistral/500samples_accuracy_standardError.png)
*caption: Standard Error for 500 samples on leo-mistral model* <br>
![500samples aya8B standardError](figures/aya8B/500samples_accuracy_standardError_AYA.png)
*caption: Standard Error for 500 samples on aya8B model* <br>
From the first two visualizations we got that there is an upwards trend of the accuracy over the buckets. To get further insides how much the slope is from 1600 to 1900, the next two visualizations were produced: <br>
![leo-mistral accuracy per year](figures/leo-mistral/500samples_years_vs_accuracy_withSlope.png)
*caption: Accuracy per year with slop and accuracy per sample as points in background for 500 samples on leo-mistral model* <br>
The visualization for the leo-mistral model shows for each template a slight positive trend in accuracy over time, though the improvements are modest. The slope values (0.02 for Template 3 and 0.03 for Templates 9 and 10) reflect minor increases in accuracy as years progress. Template 10 has consistently higher accuracy across samples and time periods compared to Templates 3 and 9. This is evident from the elevated position of the red trend line in Template 10’s plot, as well as from the wider spread of higher-accuracy points.
![aya model accuracy per year](figures/aya8B/500samples_years_vs_accuracy_withSlope_AYA.png)
*caption: Accuracy per year with slop and accuracy per sample as points in background for 500 samples on aya23 8B model* <br>
In the aya model, Template 3 and Template 9 remain at 0.03, while Template 10 has increased significantly to 0.09, suggesting that this model improves Template 10's accuracy over time more effectively than the leo-mistral model. The spread and variability of individual sample accuracies are generally similar in both visualizations. However, Template 10 in the aya model shows a higher concentration of samples achieving moderate to high accuracy. <br>

Lastly, we do not only want to know how the complete poems perform over time, but also the performance on token level per year. Therefore, we counted the number of correct predictions of each year and divided it through the number of tokens in the specific year. Applying it to the data of both models, results in the following two visualizations: <br>

![leo-mistral accuracy per token](figures/leo-mistral/500samples_accuracy_perToken.png)
*caption: Accuracy per token over the years for 500 samples on leo-mistral model* <br>
For the above seen template 3, low accuracy values are observed for most tokens across the years, with occasional small peaks, particularly around the 1750s and 1800s. The accuracy per token remains mostly under 10% throughout the entire period, with few noticeable spikes. Similar to template 3, template 9 also shows generally low accuracy, with a few sporadic peaks in specific years, notably a large spike around the early 1800s. But This spike appears to reach around 60% accuracy for tokens in that particular year, which is a significant outlier compared to the rest of the timeline. In comparison, template 10 displays a more variable trend in accuracy per token compared to the other two templates. There are several noticeable peaks, especially early in the timeline (pre-1700s) and again in the late 1800s, with accuracy values reaching around 30–40% during these periods. Additionally, A gradual increase in token accuracy toward the late 1800s is visible, suggesting that template 10 may be better suited for more recent data within the dataset. <br>

If we compare it with the results from the aya23 8B model, we can see some differences, but one similar observation.
![aya model accuracy per token](figures/aya8B/500samples_accuracy_token_aya.png)
*caption: Accuracy per token over the years for 500 samples on aya23 8B model* <br>
Starting with template 3, the spike near the 1800s is much more pronounced in this model than in the leo-mistral, reaching about 70% accuracy per token, which is a major outlier in this plot. The rest of the years fluctuate at lower accuracy levels, similar to the previous visualization. Looking at template 9 and 10 we have higher accuarcy values over time, especially a clearer upward trend in accuracy towards the 1800s to 1900s. The peak near the 1800s is also here very clearly visible. <br>

Because the peak runs across both models and is even present in all templates for one model, we decided to take a closer look at it. <br>
The analysis for the peak in the graph for accuracy per token in each year, gaves us that the corresponding year is 1782. If we go further, we come to the realization that for this exact year, only one sample exists in the complete dataset. <br>
for leo-mistral: <br>
Year: 1782 <br>
| label | POS Tags |
| ------ | ------ |
|  ADJD      |  'DOC'      |
|  APPR      |   'ADP'     |
|  ART       | 'ART'       |
|  NN        | 'NN', 'JJ'  |
|  $(        | 'ADV'       |
|  ADV       | 'ADV'       |
|  $.        | 'PUNCT'     |
|  VVFIN     | 'VVFIN'     |

for aya-model: <br>
Year: 1782 <br>

| label | POS Tags template 3 | POS Tags template 9 | POS Tags template 10 |
| ------ | ------ |------|------|
| ADJD       |  'ADJA'      | 'ADJA'        | 'ADJA'        |
| APPR       |  'APPR'      | 'APPR'        | 'APPR'        |
| ART        |  'ART'       | 'ART', 'APPR' | 'ART', 'APPR' |
| NN         |  'NN', 'ADJA' | 'NN', 'ADJA' | 'NE', 'NN'    |
| $(         |  'FM'        | 'ITJ'         | 'KOUS'        |
| ADV        |  'ADV'       | 'APPR'        | 'KOUS'        |
| $.         |  'PUN'       | '$.'          | '$.'          |
| VVFIN      |  'VVINF'     | 'VAFIN'       | 'VVFIN'       |

Compared to the whole tagset and the year before/after, 1782 has a very small number of tokens (exactly 25). The year 1778 has 1563 tokens and 1783 has 960 tokens. On the small number of tokens, we also only have 8 different labels that are represented. This makes it more likely that predictions are correct several times.

# Discussion

Based on the previous results, we can confirm the hypothesis. In both models we can clearly see the upwards trend, the further back we go. For template 10 on the aya model the slope is the highest. Additionally, the aya model shows a more defined upward trend in accuracy on token-level for Templates 9 and 10. This suggests that the model adapts better to linguistic or stylistic changes over time. <br>
Both models exhibit specific high accuracy spikes, especially for Template 10, which may imply that certain historical periods or textual characteristics are easier for the models to recognize. On the other hand, both models show high variability in accuracy across time for all templates (see in accuarcy over time with the slope). This variability may indicate that historical texts exhibit inconsistent linguistic features that challenge model consistency. <br>

Besides the results, there are also some limitations. One notable was the accuracy differences between the two models, especially since they have a similar number of parameters. There, it would be interesting to analysis whether this difference is because aya23 was trained with other languages besides German, e.g. English. Furthermore, the analysis per token showed that the data doesn't contain several poems for every year which leads to the surprising high peak. It is also not clear if in the sampled data, each year is covered. This results from the fact that the sampling was done on the buckets and not the years. <br>

As future work, it could be a good direction to build models that generalize well across such a range. This includes aspect that these models should be able to handle archaic and more modern language features without sacrificing accuracy for one over the other. Improving models for historical language understanding can significantly benefit research in fields like digital humanities, historical linguistics, and cultural studies. For example, a model that accurately interprets texts from different centuries can aid in the automated analysis of large historical corpora.

# Conclusion
The aim of the project was to analysis, how the accuarcy of the pos tagging change over time. This was done by formulating three best fitting templates and then run them on both models. The whole thing was rounded of with the anaysis of the generated data, which results in four meaningful visualizations. <br>
The analysis confirms the hypothesis that model accuracy shows an upward trend over time, with both models exhibiting improved performance as they approach more recent periods. The aya model, particularly for Template 10, demonstrates the strongest trend, suggesting a greater adaptability to linguistic and stylistic changes over time. Token-level analysis further supports this, with Templates 9 and 10 on the aya model showing distinct accuracy improvements. <br>

However, the discrepancy in accuracy between the two models, despite their similar parameter counts, suggests that the AYA model’s multilingual training might improve its adaptability with historical German texts. Additionally, uneven data distribution across years may contribute to unusual accuracy peaks, as some years are underrepresented, likely due to broader sampling buckets. <br>

Future research should focus on models that generalize well across time, handling both archaic and modern language features consistently.

# References
[1] D. I. Adelani, A. S. Doğruöz, A. Coneglian, and A. K. Ojha, “Comparing LLM prompting with cross-lingual transfer performance on indigenous and low-resource brazilian languages,” CoRR, vol. abs/2404.18286, 2024. <br>
[2] V. Aryabumi, J. Dang, D. Talupuru, S. Dash, D. Cairuz, H. Lin, B. Venkitesh, M. Smith, K. Marchisio, S. Ruder, A. Locatelli, J. Kreutzer, N. Frosst, P. Blunsom, M. Fadaee, A. Üstün, and S. Hooker. "Aya 23: Open Weight Releases to Further Multilingual Progress." (2024). <br>
[3] T. Blevins, H. Gonen, and L. Zettlemoyer, “Prompting language models for linguistic structure,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, (pp. 6649–6663). <br>
[4] “German Tagsets | Institut für Maschinelle Sprachverarbeitung | Universität Stuttgart,” Universität Stuttgart. https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/germantagsets/#id-cfcbf0a7-0 (last access:29.10.2024) <br>
[5] B. Ma, E. Nie, S. Yuan, H. Schmid, M. F¨arber, F. Kreuter, and H. Schütze, “Topro: Token-level prompt decomposition for cross-lingual sequence labeling tasks,” in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2024 - Volume 1: Long Papers, St. Julian’s, Malta, March 17-22, 2024. Association for Computational Linguistics, 2024, (pp. 2685–2702). <br>
[6] B. Plüster, C. Schuhmann, LAION, and HessianAI, “LeoLM/leo-mistral-hessianai-7b-chat · Hugging Face,” https://huggingface.co/LeoLM/leo-mistral-hessianai-7b-chat (2023). <br>
[7] E. Stüssi and P. Ströbel, “Part-of-speech tagging of 16th-century Latin with GPT,” in Proceedings of the 8th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2024), Y. Bizzoni, S. Degaetano-Ortlieb, A. Kazantseva, and S. Szpakowicz, Eds. Association for Computational
Linguistics, 2024, (pp. 196–206). <br>
[8] X. Sun, X. Li, J. Li, F. Wu, S. Guo, T. Zhang, and G. Wang, “Text classification via large language models,” in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, (pp. 8990–9005). <br>
[9] S. Wang, S. Zhuang, and G. Zuccon, “Large language models based stemming for information retrieval: Promises, pitfalls and failures,” in Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024, G. H. Yang, H. Wang, S. Han, C. Hauff, G. Zuccon, and Y. Zhang, Eds. ACM, 2024, (pp. 2492–2496).


