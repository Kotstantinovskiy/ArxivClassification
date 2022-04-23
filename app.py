import streamlit as st

from torch.nn import Softmax

from model import ArxivModel, load_model
from tokenizer import get_tokenizer

from lables import num_to_classes, taxonomy

from parser import get_text_title

model = load_model()
tokenizer = get_tokenizer()

arxiv_model = ArxivModel(model, tokenizer)
softmax = Softmax(dim=1)

st.markdown("### Classification of article topics")

col1, col2 = st.columns(2)
text = ""

frame = "text"
option = st.selectbox('Choose to write title yourself or input url/id of article',
                      ('Write the title and summary yourself',
                       'Input url or id of article'))

if option == 'Write the title and summary yourself':
    frame = 'text'
else:
    frame = 'url'

if frame == 'text':
    title_text = st.text_input("Write title of article", key='arxiv_title_input')
    summary_text = st.text_area("Write summary of article (optional)", key='arxiv_sum_input')
    click_button_text = st.button('Submit', key=1)

    if click_button_text and title_text.strip() == "":
        text = ""
        if summary_text.strip() != "":
            st.markdown(f'<p style="color:#FF2D00;font-size:18px">Please, input title</p>', unsafe_allow_html=True)
    elif click_button_text and title_text.strip() != "" and summary_text.strip() != "":
        text = title_text.strip() + '\t' + summary_text.strip()
    elif click_button_text and title_text.strip() != "":
        text = title_text.strip()
    text = text.strip()

elif frame == 'url':
    id_url = st.text_input("Write article's url or id", key='arxiv_id_input').strip()
    click_button_url = st.button('Submit', key=1)

    if click_button_url and id_url != "":
        res = get_text_title(id_url)
        if res is not None:
            text = res[0].strip() + '\t' + res[1].strip()
            text = text.strip()
        else:
            st.markdown(f'<p style="color:#FF2D00;font-size:18px">Incorrect url or id</p>', unsafe_allow_html=True)
            text = ""

if text.lower() == 'i want a cake':
    st.markdown("# :cake:")
elif text != "":
    idxs = arxiv_model.get_idx_class(text, thr=0.95)
    print(len(idxs))
    if len(idxs) > 80:
        st.markdown("#### Sorry, model can't classify the article with high confidence")
    else:
        idxs = idxs[:10]
        st.markdown("#### The model have defined:")
        for idx, prob in idxs:
            if taxonomy.get(num_to_classes[idx], -1) != -1:
                st.markdown("{} \t {}%".format(taxonomy.get(num_to_classes[idx], -1), round(prob * 100, 1)))
else:
    st.markdown("")
