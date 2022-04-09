from data_process import data_process
from pyecharts.charts import WordCloud
from pyecharts import options as opts
import json

clean_data, labels, data_after_stop = data_process()


def word_fre_counts(num):
    word_dict = {}
    for i in data_after_stop[labels == num]:
        for j in i:
            if j not in word_dict.keys():
                word_dict[j] = 1
            else:
                word_dict[j] += 1

    # convert dictionary to list of tuples
    word_fre_count = [(k, v) for k, v in word_dict.items()]

    return word_fre_count


spam_word_fre = word_fre_counts(1)
non_spam_word_fre = word_fre_counts(0)


def word_cloud(text):

    cloud = (
        WordCloud()
        .add("",
             text,
             word_size_range=[12, 100],
             textstyle_opts=opts.TextStyleOpts(font_family="Kai"))
    )

    return cloud


spam_word_cloud = word_cloud(spam_word_fre)
non_spam_word_cloud = word_cloud(non_spam_word_fre)

spam_word_cloud.render("./results/spam_word_cloud.html")
non_spam_word_cloud.render("./results/non_spam_word_cloud.html")
