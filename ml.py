import os
import numpy as np
import fitz
import urllib.request
from tqdm import tqdm

# ML
from transformers import pipeline

# Local files
from config import articles_to_cache

# Supress warnings
import warnings
warnings.filterwarnings("ignore")


# Setup
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")


def get_references(prefix_path, arxiv_id):
    article_folder = os.path.join(prefix_path, arxiv_id)
    article_pdf = os.path.join(article_folder, "article.pdf")

    # Parse text from article
    with fitz.open(article_pdf) as doc:  # open document
        sentences = "\n".join([page.get_text().lower() for page in doc])

    sentences = sentences.split('\n')

    # Parse all lines in format "[x] ..."
    references = []
    for ind, sentence in enumerate(sentences):
        if "[" in sentence and len(sentence.split("[")[1]) > 1 and "]" in sentence.split("[")[1]:
            text_in = sentence.split("[")[1].split("]")[0]
            if text_in.isnumeric() and (sentence.split("[")[0].isspace() or sentence.split("[")[0] == ""):
                references.append([int(text_in), sentence])

        elif len(references) > 0:
            references[-1][1] += " " + sentence
            

    # Reverse references and iterate from end until 1
    references = references[::-1]
    last = None
    reference_arxiv_ids = []
    for ref in references:
        if last is not None and last != ref[0] + 1:
            break

        arxiv_split = ref[1].split("arxiv:")
        abs_split = ref[1].split("abs/")

        arxiv_id = ""
        if len(arxiv_split) > 1:
            for s in arxiv_split[1]:
                if not (s.isnumeric() or s == "."):
                    break
                arxiv_id += s

        elif len(abs_split) > 1:
            for s in abs_split[1]:
                if not (s.isnumeric() or s == "."):
                    break
                arxiv_id += s

        print(ref[0], arxiv_id)
        reference_arxiv_ids.append(arxiv_id)
        last = ref[0]

    return reference_arxiv_ids[::-1]


def cache_article(prefix_path, arxiv_id, article_name="None", eps=2, step_size=2):
    article_folder = os.path.join(prefix_path, arxiv_id)
    if os.path.exists(article_folder):
        print(f"Article {article_name}:{arxiv_id} already cached")
        return

    print(f"Starting summarisation of article \"{article_name}\":{arxiv_id}")
    os.mkdir(article_folder)

    # Download and parse pdf
    article_pdf = os.path.join(article_folder, "article.pdf")
    urllib.request.urlretrieve(f"https://arxiv.org/pdf/{arxiv_id}.pdf", article_pdf)
    with fitz.open(article_pdf) as doc:  # open document
        text = "\n".join([page.get_text().lower() for page in doc])

    sentences = text.replace('\n', ' ').split('. ')

    # Prepare summaries
    summaries = []
    for ind in tqdm(range(0, len(sentences), step_size)):
        sent_slice = sentences[max(0, ind - eps):min(len(sentences) - 1, ind + eps)]
        if len(". ".join(sent_slice)) < 100:
            continue

        summary = summarizer(". ".join(sent_slice))[0]["summary_text"]
        summaries.append(summary)

    with open(os.path.join(article_folder, f'summaries.npy'), 'wb') as f:
        np.save(f, np.array(summaries))

    print("Cache completed")


def cache_articles():
    print("Start article cache")
    cache_path = os.path.join("static", "cache")

    for article_name in articles_to_cache:
        arxiv_id = articles_to_cache[article_name]
        cache_article(cache_path, arxiv_id, article_name)
        
        ref_ids = get_references(cache_path, arxiv_id)[:2]
        for ref_arxiv_id in ref_ids:
            if ref_arxiv_id != "":
                cache_article(cache_path, ref_arxiv_id)

