import pandas as pd
import wordcloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud


class JupyterService:

    def __init__(self, context):
        self.context = context
        self.df = None

    def get_by_sql(self, sql):
        if sql.strip().lower().startswith("select"):
            with self.context.engine.connect() as conn:
                self.df = pd.read_sql(sql, conn)
            return self.df
        return "Only for select statement"

    def get_topics(self, source: str, question_nb: int):
        sql = f"""select topic.label as topic, count(form_topic.id) as nb_form from topic
            join form_topic on form_topic.topic_id=topic.id
            where source='{source}'
            and question_nb={question_nb}
            group by topic.id
            having count(form_topic.id) > 1
            order by nb_form desc"""
        df = self.get_by_sql(sql)
        return df

    def get_stats(self, source: str):
        sql = f"""SELECT avg(count) as average, 
         PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY count) as median_,
         PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY count) as quartile, 
         PERCENTILE_CONT(0.1) WITHIN GROUP(ORDER BY count) as decile, 
         PERCENTILE_CONT(0.01) WITHIN GROUP(ORDER BY count) as percentile 
         FROM topic where source='{source}' and count > 1"""
        df = self.get_by_sql(sql)
        return df

    def get_lemas(self, topic: str, source: str):
        sql = f"""select topic.label as topic_label, lema.label as lema_label, lema.count from topic
        join lema on lema.topic_id=topic.id
        where topic.label = '{topic}'
        and source='{source}'
        order by lema.count desc"""
        df = self.get_by_sql(sql)
        return df

    def show_lemas(self, df):
        text = ""
        for index, row in df.iterrows():
            text += row.lema_label + " "
        text = text.replace("amp", "")
        wordcloud = WordCloud(max_font_size=50, max_words=20, background_color="grey").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

    def get_scores(self, source: str, question_nb: int, empathy_category: str, positive: bool, denominator_thresold: float, numerator_thresold: float, debug=False, exp_max=10e9):
        category = 0
        category_inverse = 2
        category_term = "negative"
        category_inverse_term = "positive"
        if positive:
            category = 2
            category_inverse = 0
            category_term = "positive"
            category_inverse_term = "negative"
        formula = "count(form_topic.id)::float/coalesce(sub_topic.nb_sub_form+1, 0.1)"
        aggregate = "count(form_topic.id)"
        score_term = "nb"
        if source == "tdidf":
            aggregate = "sum(form_topic.score)"
            score_term = "score"
            formula = "sum(form_topic.score)::float/coalesce(sub_topic.nb_sub_form+1, 0.1)"
        sql = f"""select topic.id, topic.label as topic, {aggregate} as {score_term}_{category_term}_form, sub_topic.nb_sub_form as {score_term}_{category_inverse_term}_form, {formula} as score from topic
        join form_topic on form_topic.topic_id=topic.id
        join form on form_topic.form_id=form.id
        join stat on stat.id=form.id
        left join (select topic.id, {aggregate} as nb_sub_form from topic
        	join form_topic on form_topic.topic_id=topic.id
        	join form on form_topic.form_id=form.id
        	join stat on stat.id=form.id
        	where source='{source}'
        	and stat.{empathy_category}_category={category_inverse}
            and form_topic.question_nb={question_nb}
        	group by topic.id
        	having {aggregate} > {denominator_thresold} and {aggregate} < {exp_max}
        ) sub_topic on sub_topic.id = topic.id
        where source='{source}'
        and stat.{empathy_category}_category={category}
        and form_topic.question_nb={question_nb}
        group by topic.id, sub_topic.nb_sub_form
        having {aggregate} > {numerator_thresold}
        order by score desc,  {score_term}_{category_term}_form desc"""
        if debug:
            print(sql)
        df = self.get_by_sql(sql)
        return df




