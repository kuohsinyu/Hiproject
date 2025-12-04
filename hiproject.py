import os
import json
import time
import random
import re

import googleapiclient.discovery
import jieba
import jieba.analyse
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==============================
#  環境設定
# ==============================

# 防止 OpenMP 衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

# 檔案與路徑設定
MODEL_SAVE_PATH = "./trained_model"
TRAIN_DATA_PATH = "trainword.json"
TAIWAN_MASK_PATH = "taiwan.jpg"
FONT_PATH = "AdobeFanHeitiStd-Bold.otf"  # 請確認檔案存在於專案資料夾


# ==============================
#  YouTube API 相關
# ==============================

def get_youtube_client(api_key: str):
    """初始化 YouTube API Client。"""
    return googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)


def get_comment_replies(parent_id: str, youtube):
    """抓取單一頂層留言的所有回覆。"""
    replies = []
    next_page_token = None

    while True:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText",
        )
        response = request.execute()

        for item in response["items"]:
            reply = item["snippet"]["textDisplay"]
            replies.append(reply)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return replies


def get_video_comments(video_id: str, api_key: str):
    """抓取指定影片的所有留言（含回覆）。"""
    youtube = get_youtube_client(api_key)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText",
        )
        response = request.execute()

        for item in response["items"]:
            top_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(top_comment)

            # 抓取頂層留言的所有回覆
            total_reply_count = item["snippet"]["totalReplyCount"]
            if total_reply_count > 0:
                parent_id = item["id"]
                replies = get_comment_replies(parent_id, youtube)
                comments.extend(replies)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


# ==============================
#  NLP 前處理與關鍵詞抽取
# ==============================

def keep_chinese_chars(text: str) -> str:
    """只保留中文，去掉其他符號／英文。"""
    pattern = re.compile(r"[^\u4e00-\u9fff]")
    chinese_text = re.sub(pattern, "", text)
    return chinese_text


def segment_comments(comments):
    """使用 jieba 對留言進行分詞。"""
    return [" ".join(jieba.lcut(comment)) for comment in comments]


def extract_keywords(text: str, top_k: int = 10):
    """先只保留中文，再用 jieba.analyse 抽關鍵詞。"""
    text = keep_chinese_chars(text)
    return jieba.analyse.extract_tags(text, topK=top_k)


# ==============================
#  模型訓練與推論（BERT）
# ==============================

def train_model(train_texts, train_labels, model_save_path: str):
    """訓練中文 BERT 情緒分類模型。"""
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

    print("Tokenizing the data...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=512,
    )

    train_dataset = Dataset.from_dict(
        {
            "input_ids": train_encodings["input_ids"],
            "attention_mask": train_encodings["attention_mask"],
            "labels": train_labels,
        }
    )

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    print("Saving model...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return model, tokenizer


def load_model(model_path: str):
    """從指定路徑載入已訓練好的模型與 tokenizer。"""
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict_emotions(model, tokenizer, comments):
    """對留言進行情緒預測。"""
    inputs = tokenizer(
        comments,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()
    return predictions


# ==============================
#  視覺化：文字雲
# ==============================

def create_wordcloud(text: str, mask_image_path: str, output_image_path: str, font_path: str):
    """產生文字雲圖片。"""
    if not text.strip():
        print(f"[WARN] Text is empty, skip generating wordcloud: {output_image_path}")
        return

    mask = np.array(Image.open(mask_image_path))
    wordcloud = WordCloud(
        font_path=font_path,
        width=800,
        height=500,
        background_color="white",
        mask=mask,
    ).generate(text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved wordcloud to {output_image_path}")


# ==============================
#  主流程
# ==============================

def main():
    print("=== YouTube 中文情緒分析程式啟動 ===")
    t_start = time.time()

    # 1. 取得 API Key（從環境變數，而不是寫死在程式裡）
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到環境變數 YOUTUBE_API_KEY，請先在終端機設定：\n"
            "export YOUTUBE_API_KEY='你的_API_KEY'"
        )

    # 2. 取得影片 URL，解析 video_id
    video_url = input("請輸入影片 URL: ").strip()
    if "v=" not in video_url:
        raise ValueError("URL 格式看起來不對，無法解析出 video_id（缺少 'v=' 參數）")

    video_id = video_url.split("v=")[1].split("&")[0]
    print(f"[INFO] 解析到的 video_id: {video_id}")

    # 3. 爬取留言
    print("[STEP] 爬取 YouTube 留言中...")
    comments = get_video_comments(video_id, api_key)
    print(f"[INFO] 共取得 {len(comments)} 則留言與回覆。")

    # 4. 斷詞與關鍵詞抽取
    print("[STEP] 進行斷詞與關鍵詞抽取...")
    segmented_comments_raw = segment_comments(comments)

    segmented_comments = []
    for c in tqdm(segmented_comments_raw, desc="Extracting keywords"):
        keywords = extract_keywords(c)
        keywords_str = ",".join(keywords)
        segmented_comments.append(keywords_str)

    print(f"[INFO] 關鍵詞處理後留言數：{len(segmented_comments)}")

    # 5. 模型載入或訓練
    print("[STEP] 檢查是否已有訓練好的模型...")
    if os.path.exists(MODEL_SAVE_PATH):
        print("[INFO] 發現已存在的模型，載入中...")
        model, tokenizer = load_model(MODEL_SAVE_PATH)
    else:
        print("[INFO] 未找到模型，準備開始訓練新模型...")
        if not os.path.exists(TRAIN_DATA_PATH):
            raise FileNotFoundError(
                f"找不到訓練資料 {TRAIN_DATA_PATH}，請確認檔案存在或修改路徑。"
            )

        train_texts = []
        train_labels = []

        with open(TRAIN_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)

        for sample in tqdm(data, desc="Processing training data"):
            for text, label in sample:
                # 可以在這裡決定是否要做 downsampling / upsampling
                keywords = extract_keywords(text)
                keywords_str = ",".join(keywords)
                train_texts.append(keywords_str)
                train_labels.append(int(label))

        print(f"[INFO] 訓練資料筆數：{len(train_texts)}")
        model, tokenizer = train_model(train_texts, train_labels, MODEL_SAVE_PATH)

    # 6. 預測情緒
    print("[STEP] 進行情緒預測...")
    predictions = predict_emotions(model, tokenizer, segmented_comments)

    # 7. 根據預測結果分成正面與負面
    print("[STEP] 分類正面與負面留言...")
    positive_comments = []
    negative_comments = []

    for comment, prediction in zip(segmented_comments, predictions):
        # 按照你的原邏輯：1,5 當作正面；2,3,4 當作負面
        if prediction in [1, 5]:
            positive_comments.append(comment)
        elif prediction in [2, 3, 4]:
            negative_comments.append(comment)

    print(f"[INFO] 正面留言數：{len(positive_comments)}")
    print(f"[INFO] 負面留言數：{len(negative_comments)}")

    # 8. 產生文字雲
    print("[STEP] 生成文字雲圖片...")
    positive_text = " ".join(positive_comments)
    negative_text = " ".join(negative_comments)

    create_wordcloud(positive_text, TAIWAN_MASK_PATH, "positive_wordcloud.png", FONT_PATH)
    create_wordcloud(negative_text, TAIWAN_MASK_PATH, "negative_wordcloud.png", FONT_PATH)

    # 9. 結束計時
    t_end = time.time()
    print(f"=== 完成！總執行時間：{t_end - t_start:.2f} 秒 ===")


if __name__ == "__main__":
    main()
