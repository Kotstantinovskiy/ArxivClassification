import arxiv


def get_text_title(arxiv_id):
    if 'arxiv.org' in arxiv_id:
        arxiv_id = arxiv_id.split('/')[-1]

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return paper.title, paper.summary
    except arxiv.UnexpectedEmptyPageError:
        return None
    except arxiv.HTTPError:
        return None
