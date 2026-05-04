# import traceback
# import gradio as gr

# from i18n.i18n import I18nAuto


# def create_faq_tab(i18n: I18nAuto):
#     tab_faq = i18n("FAQ")
#     with gr.TabItem(tab_faq):
#         try:
#             with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
#                 info = f.read()
#             gr.Markdown(value=info)
#         except:
#             gr.Markdown(traceback.format_exc())
