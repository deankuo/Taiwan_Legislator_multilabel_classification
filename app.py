import gradio as gr
from transformers import pipeline
import pandas as pd
import time

markdown_content = """
## Legislator Interpellation Classifier
This is a classifier for legislator dialogues. Please enter the dialogue text in the input box below, and the model will predict its label.
Features:
- Trained using the transcript records of Taiwanese city councilors.
- Multi-label task.
- 4 types of Labels including: 要求資訊、要求說明、要求改變、威脅制裁.
We used the following model [dean22029/Taiwan_Legislator_multilabel_classification](https://huggingface.co/dean22029/Taiwan_Legislator_multilabel_classification)
"""


# 加載預訓練模型
model = pipeline("text-classification", model="dean22029/Taiwan_Legislator_multilabel_classification", top_k=None)

label_mapping = {'LABEL_0': '要求資訊', 'LABEL_1': '要求說明', 'LABEL_2': '要求改變', 'LABEL_3': '威脅制裁'}

def predict_label(text, progress=gr.Progress(), max_len=510):
    # Take the last 512 tokens
    truncated_text = text[-max_len:]
    progress(0, desc="開始處理")
    time.sleep(1)
    progress(0.3, desc=f"正在分析文本")
    predictions = model(truncated_text)
    
    # Prediction
    labels = [label_mapping.get(prediction['label']) for prediction in predictions[0] if prediction['score'] > 0.5]
    label_scores = {label_mapping.get(prediction['label']): prediction['score'] for prediction in predictions[0]}

    # Output Dataframe
    df = pd.DataFrame(label_scores.items(), columns=['Label', 'Probability'])
    
    return ", ".join(labels) if labels else "無相關標籤", df

interface = gr.Interface(
    fn=predict_label, 
    inputs=gr.Textbox(lines=2, placeholder="輸入您想分析的文字..."), 
    outputs=["label", "dataframe"],
    theme="Default",
    title="Legislator interpellation Classification",
    description=markdown_content,
    allow_flagging="never",
    examples=[["韓國瑜，本席直接問你，站在你旁邊的市長，是不是就是你說的「男人世界裡面的『豎仔』（臺語）」！沒有擔當！"],
              ["這個問題我有問過交通局，交通局說依照規定，他們只能劃設2公尺至8公尺的斑馬線，而且只能畫在人行道，騎樓是不能畫設。局長，是不是？"],
              ["你不用跟本席講這個風涼話！什麼優點？營運績效？市長，臺北農產運銷公司抽取固定的營業額比例，另外還有每年1億2,000萬元的停車場業外收入，這樣的工作是「孤行獨市」（臺語），臺北農產運銷公司是「孤行獨市」，張三、李四、王二麻子來做，也都會賺錢啦！你懂嗎？結果你找個流氓來做！"],
              ["柯市長我..我告訴你，我告訴你，我們選民選你是當市長不是選你當皇帝，你現在 藉由這樣網路，你操控的網軍操作，對於監督你的、批評你的，極盡網路霸凌之能事，柯市長，你的網軍現在是齁，順柯者生啦，逆柯者亡，這樣的行為、做法，跟納粹時期啊那個黑衫軍一樣的!柯市長，過去台灣的民主運動你沒有參與我不怪你，可是，你用這種新型態的網路霸凌、操控網軍來挑起世代對立，來傷害我們台灣的民主體制，這一點，我沒辦法原諒你，我瞧不起你!"],
              ["你不用...哩母面墊底遐台上沒幾分鐘哩咧五四三，證據擺的很清楚，旁邊這位我不 想看到你(大熊)，你下去，誰叫你上來的。 證據是甚麼?你把790萬的市府預算，你拿去做你的網路宣傳影片，所有的網紅，蘋果C打 啦黑素斯啦上班不要看啦啾啾鞋啦甚麼阿滴英文，所有，哦，最近你還要做的台客劇場、 呱吉、蔡阿嘎等等我想你一定心裡有數嘛，你怎麼會不曉得呢?這些，如果你是拿你去抵押的房產，欸，你拿來做，那你說這個是個人，你的政治宣傳，那我們沒話講，你公然堂而皇之的拿公帑來拍攝網路影片，宣揚你自己，而且裡面談的，其實跟市政風馬牛不相干，在我看哪，柯市長我不客氣的講，你就是沒本事才會搞這種東西"],
              ["局長，我瞭解了。之後如果還有什麼細節要報告，可以到我的辦公室來。今天這段總質詢的時間，通常我不喜歡問個案的問題，但是今天這個個案，如果沒有被妥善處理的話，也會變成臺北市教育局系統性的問題，會讓臺北市的家長開始不信任臺北市教育局針對這種不當體罰案件的處理，而且實質上這些家長跟我抱怨，但這個抱怨目前我也還沒證實，就是他們說後來有在教育局的貼文上表達他們的不滿和抗議，但是卻被隱蔽了他們的發言內容。"],
              ["接著這個議題也跟空污有關係，在林園石化工業區旁邊有中芸國小，中芸國小有架設，這應該是環保署所架設的FTIR光譜分析檢測儀，要檢測空污排放的數據，這些的檢測都超過周界的標準，都是有毒的化學物質。這一點當初在雲林縣麥寮有一間許厝國小，那一間就是由台塑全部來幫學童做一個健康風險評估，結果才發現小朋友血液中的濃度有一些都超標，所以不得不把那所學校（許厝分校）遷移，所以在林園、大發工業區國小的學童應該也要來做這一些健康風險評估的檢查，這樣比較了解啊!因為在整個林園、大寮地區目前健康風險評估、流行病學的一個研究，目前還沒有做。大林蒲六個里都有做，所以才啟動遷村，為什麼啟動遷村?就是因為經過健康風險評估、流行病學調查，發現大林蒲那個地方的危害物質真的太高，所以一定要遷村。這一份是由我們國衛院副研究員陳裕政所調查出來的，他說大林蒲地區的環境監測結果與國衛院對其他地區正進行調查的地區相比較，大林蒲地區的危害物質濃度低於林園工業區，比雲林六輕高，這一點我請問衛生局長，國衛院提出的這一份，市政府衛生局知道嗎?局長，請答復。這個意思就表示，林園工業區危害物質的濃度比大林蒲還高，那麼大林蒲要遷村，現在這個問題就變成林園怎麼辦?局長，那是一般的健康檢查，本席所說的是做流行病學研究和健康風險評估，這個要比較專業，而且也比較針對一些有害的化學物質的化學成分來做研究。所以這個不管由市政府來啟動，看是要用我們的空污基金，或者是要由當地的石化工業區、或者是工業局、或者是市政府的財源，既然國衛院已經提出大林蒲地區的危害物質濃度低於林園工業區，那麼林園就變成全高雄市危害物質濃度最高的地區，比大林蒲還高，所以這個議題不能就這樣打住，不是只有大林蒲遷村，遷村以後，那林園怎麼辦?大林蒲地區所有的不管是那些重工業、重污染有沒有改善?會不會飄散到林園?不是只有把大林蒲這六個里遷走，然後眼不見為淨，那麼那裡的污染要怎麼辦?污染要跑去哪裡?現在只剩下林園，再來就由林園接收，林園那些百姓的生命真是在旦夕之間，真的是很危險。所以我們一定嚴格要求，市政府要啟動林園地區的健康風險評估、流行病學調查。這一點請市長答復一下，市長，請答復。韓市長，剛剛提到的是針對國小學童，但本席認為我們有幾個村落跟工業區是鄰近的、一牆之隔的，那些村落大概有8個里，那8個里也要一併來做調查，不是只有國小，而要包括8個里，24個里裡面有8個里跟工業區是一牆之隔的。"]]
)

interface.launch(debug=True, share=True, show_error=True)