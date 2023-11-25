import os
import math
import numpy as np
import fitz
import urllib.request
from tqdm import tqdm
from colour import Color

# ML
from transformers import pipeline
import torch

# Local files
from config import articles_to_cache, highlight_n, topk, k, eps_input, eps_ref

# Supress warnings
import warnings
warnings.filterwarnings("ignore")


# Setup
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def cache_article(prefix_path, arxiv_id, article_name="None", step_size=2):
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
        sent_slice = sentences[max(0, ind - eps_ref):min(len(sentences) - 1, ind + eps_ref)]
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

        ref_ids = get_references(cache_path, arxiv_id)
        for r_ind, ref_arxiv_id in enumerate(ref_ids):
            if ref_arxiv_id != "":
                print(f"{r_ind + 1} / {len(ref_ids)}")
                cache_article(cache_path, ref_arxiv_id)


def predict_document(arxiv_id, cache_path, pdf_folder):
    # Create output directory
    highlight_path = os.path.join(pdf_folder, arxiv_id)
    if os.path.exists(highlight_path):
        return
    os.mkdir(highlight_path)


    all_doc_references = get_references(cache_path, arxiv_id)
    all_line_ref_indexes = []
    for reference_number in range(1, len(all_doc_references) + 1):
        print(f"Starting reference {reference_number} / {len(all_doc_references)}")
        all_line_ref_indexes.append(list())
        arxiv_ref_id = all_doc_references[reference_number - 1]
        if arxiv_ref_id == "":
            print("No link")
            continue  # No link found

        ref_cache_path = os.path.join(cache_path, arxiv_ref_id)

        document_ref = os.path.join(ref_cache_path, "article.pdf")
        doc_path = os.path.join(cache_path, arxiv_id, "article.pdf")

        from sentence_transformers import SentenceTransformer
        model_paperswithcode_word2vec = SentenceTransformer('lambdaofgod/paperswithcode_word2vec', device="cuda")

        red = Color("red")
        grad_colors = list(red.range_to(Color("yellow"), 11))

        # Read pdf's sentences
        with fitz.open(doc_path) as doc:  # Open document
            text_ref = "\n".join([page.get_text().lower() for page in doc])
        sentences = text_ref.replace('\n', ' ').split('. ')

        with fitz.open(document_ref) as doc:  # open document
            text = "\n".join([page.get_text().lower() for page in doc])
        sentences_ref = text.replace('\n', ' ').split('. ')

        # Read cached summaries
        summaries_on_ref = np.load(os.path.join(ref_cache_path, "summaries.npy"))

        for ind, sentence in enumerate(sentences):
            if "[" not in sentence:
                continue
            
            # Iterate over all [x] in current row and search for the one we need
            found_reference = 0
            for elem in sentence.split("[")[1:]:
                if "]" not in elem:
                    continue

                text_in = elem.split("]")[0]
                if text_in.isnumeric() and int(text_in) == reference_number:
                    found_reference += 1

            if found_reference > 0:
                # Check input window size
                print("--", sentence, "--")
                main_sent_embedding = model_paperswithcode_word2vec.encode(sentence)

                sent_dists = []
                for s in sentences[max(0, ind - eps_input):min(len(sentences) - 1, ind + eps_input)]:
                    curr_embedding = model_paperswithcode_word2vec.encode(s)
                    d = np.linalg.norm(main_sent_embedding - curr_embedding)
                    sent_dists.append(d)

                sent_indexes = np.argsort(sent_dists)
                sent_slice = []
                for chosen_ind in sent_indexes[:topk]:
                    sent_slice.append((
                        max(0, ind - eps_input) + chosen_ind,
                        sentences[max(0, ind - eps_input) + chosen_ind]
                    ))

                sent_slice = [elem[1] for elem in sorted(sent_slice)]

                avg = sum(sent_dists) / len(sent_dists)
                median = np.median(np.array(sent_dists))

                # Count input window's embedding
                # print(text_in, ind, "||", sent_slice, '\n')
                summary = summarizer(". ".join(sent_slice), max_length=100)[0]["summary_text"]
                print(reference_number, "SUMMARY: ", summary)
                embedding = model_paperswithcode_word2vec.encode(summary)


                # Iterate over reference paper
                s_ind = 0
                dists = []
                ind_map = dict()
                for ind_ref in range(0, len(sentences_ref), 2):
                    sent_ref = sentences_ref[ind_ref]
                    sent_slice_ref = sentences_ref[max(0, ind_ref - eps_ref):min(len(sentences_ref) - 1, ind_ref + eps_ref)]

                    if len(". ".join(sent_slice_ref)) < 100:
                        continue

                    summary_ref = summaries_on_ref[s_ind]
                    s_ind += 1
                    embedding_ref = model_paperswithcode_word2vec.encode(summary_ref)

                    dist = np.linalg.norm(embedding - embedding_ref)
                    ind_map[len(dists)] = ind_ref
                    dists.append(dist)


                indices = np.argsort(dists)
                selected_inds = list()
                best_conf = dists[indices[0]]

                # Highlight example
                with fitz.open(document_ref) as doc:  # open document
                    for close_ind in indices:
                        # Print only distances that are far from each other in text
                        sents = np.abs(np.asarray(selected_inds) - close_ind)
                        if len(sents) != 0:
                            min_dist = sents.min()
                            if min_dist < 5:
                                continue

                        slice_ans = sentences_ref[max(0, ind_map[close_ind] - eps_ref):min(len(sentences_ref) - 1, ind_map[close_ind] + eps_ref)]
                        selected_inds.append(close_ind)
                        loaded_text = "\n".join([page.get_text() for page in doc]).replace('\n', ' ').replace("- ", "").split('. ')
                        all_sentences = loaded_text[max(0, ind_map[close_ind] - eps_ref):min(len(sentences_ref) - 1, ind_map[close_ind] + eps_ref)]

                        conf = dists[close_ind]
                        color_ind = min(10, int(math.sqrt(conf - best_conf) * 6))
                        # print(dists[close_ind], color_ind)

                        for highlight_text in all_sentences:
                            for page in doc:
                                text_instances = page.search_for(highlight_text)
                                if text_instances is None:
                                    continue

                                # Highlight
                                for inst in text_instances:
                                    highlight = page.add_highlight_annot(inst)
                                    highlight.set_colors({"stroke": grad_colors[color_ind].rgb})
                                    highlight.update()


                        if len(selected_inds) == highlight_n:
                            doc.save(os.path.join(highlight_path, f"{arxiv_ref_id}_{ind}.pdf"), garbage=4, deflate=True, clean=True)
                            for i in range(found_reference):
                                all_line_ref_indexes[reference_number - 1].append(ind)
                            break

                    print()


    # Insert link
    with fitz.open(doc_path) as main_doc:  # open document
        for reference_number in range(1, len(all_doc_references) + 1):
            arxiv_ref_id = all_doc_references[reference_number - 1]
            if arxiv_ref_id == "":
                continue

            occurance_ind = 0
            for page_index, page in enumerate(main_doc):
                # all_links = page.get_links()
                # for link in all_links:
                #     page.delete_link(link)

                text_instances = page.search_for(f"[{reference_number}]")
                for inst in text_instances:
                    uri = f"http://papers.pavtiger.com/pdf/{arxiv_id}/{arxiv_ref_id}_{all_line_ref_indexes[reference_number - 1][occurance_ind]}.pdf"
                    print(uri)
                    page.insert_link({"kind": fitz.LINK_URI, 'from': fitz.Rect(inst.x0, inst.y0, inst.x1, inst.y1), "uri": uri})
                    occurance_ind += 1

        main_doc.save(os.path.join(highlight_path, f"main.pdf"), garbage=4, deflate=True, clean=True)

