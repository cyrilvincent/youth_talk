import numpy as np
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

    def get_lemas(self, topic: str, source: str, debug=False):
        if source == "tdidf":
            source = "textrank"
        sql = f"""select topic.label as topic_label, lema.label as lema_label, lema.count from topic
        join lema on lema.topic_id=topic.id
        where topic.label = '{topic}'
        and source='{source}'
        order by lema.count desc"""
        df = self.get_by_sql(sql)
        if debug:
            print(sql)
        if len(df) == 0:
            df = self.get_lemas_2(topic, source)
        return df

    def get_lemas_2(self, lema: str, source: str):
        sql = f"""select topic.label as topic_label, lema2.label as lema_label, lema2.count from lema
        join topic on lema.topic_id=topic.id
        join lema as lema2 on lema2.topic_id=topic.id
        where lema.label = '{lema}'
        and source='{source}'
        order by lema2.count desc"""
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

    def get_scores(self, source: str, question_nb: int, empathy_category: str, positive: bool, denominator_thresold: float, numerator_thresold: float, debug=False, exp_max=10e9, sentiment=False, gpt_comment=False):
        if empathy_category == "empathy":
            gpt_comment = False
        category = 0
        category_inverse = 2
        category_term = "negative"
        category_inverse_term = "positive"
        if positive:
            category = 2
            category_inverse = 0
            category_term = "positive"
            category_inverse_term = "negative"
        formula = "count(form_topic.id)::float/coalesce(sub_topic.nb_sub_form, 0.1)"
        aggregate = "count(form_topic.id)"
        score_term = "nb"
        if source == "tdidf":
            aggregate = "sum(form_topic.score)"
            score_term = "score"
            formula = "sum(form_topic.score)::float/coalesce(sub_topic.nb_sub_form, 0.1)"
        q_sql = "q" + str(question_nb)[0] + "_" + str(question_nb)[1]
        sentiment_positive = f"and stat.{q_sql}_sentiment < -0.33 "
        sentiment_negative = f"and stat.{q_sql}_sentiment > 0.33 "
        if positive:
            sentiment_negative, sentiment_positive = sentiment_positive, sentiment_negative
        gpt_select = ""
        gpt_where = ""
        if gpt_comment:
            gpt_select = ", gpt_comment.comment as explaination"
            gpt_where = f"""left outer join gpt_comment on gpt_comment.topic_id=topic.id and gpt_comment.question_nb={question_nb} 
                and gpt_comment.positive is {str(positive).upper()}
                and gpt_comment.empathy='{empathy_category}'"""
        sql = f"""select topic.id, topic.label as topic, {aggregate} as {score_term}_{category_term}_form, sub_topic.nb_sub_form as {score_term}_{category_inverse_term}_form, {formula} as score {gpt_select} from topic
        join form_topic on form_topic.topic_id=topic.id
        join form on form_topic.form_id=form.id
        join stat on stat.id=form.id
        left join (select topic.id, {aggregate} as nb_sub_form from topic
        	join form_topic on form_topic.topic_id=topic.id
        	join form on form_topic.form_id=form.id
        	join stat on stat.id=form.id
        	where source='{source}'
        	and stat.{empathy_category}_category={category_inverse}
        	{sentiment_negative if sentiment else ""}
            and form_topic.question_nb={question_nb}
        	group by topic.id
        	having {aggregate} > {denominator_thresold} and {aggregate} < {exp_max}
        ) sub_topic on sub_topic.id = topic.id
        {gpt_where}
        where topic.source='{source}'
        and stat.{empathy_category}_category={category}
        and form_topic.question_nb={question_nb}
        {sentiment_positive if sentiment else ""}
        group by topic.id, sub_topic.nb_sub_form {", gpt_comment.comment" if gpt_comment else ""}
        having {aggregate} > {numerator_thresold}
        order by score desc,  {score_term}_{category_term}_form desc"""
        if debug:
            print(sql)
        df = self.get_by_sql(sql)
        return df

    def get_gpt_comment(self, topic: str, question: int, empathy: str, positive: bool):
        sql = f"""select topic.label as topic, gpt_comment.comment as explaination from topic
        left join gpt_comment on gpt_comment.topic_id=topic.id
            and gpt_comment.question_nb={question}
            and gpt_comment.empathy='{empathy}'
            and gpt_comment.positive is {str(positive).upper()}
        where topic.label = '{topic}'
        and topic.source='gpt-4o-mini'
        """
        df = self.get_by_sql(sql)
        return df

    def get_highlevel(self, question: int, empathy: str, positive: bool):
        column = empathy + ("2" if positive else "0")
        sql = f"""select chatgpt_highlevel.id, chatgpt_highlevel.index as main_topic, chatgpt_highlevel.{column}::float/100 as score from chatgpt_highlevel
        where chatgpt_highlevel.question_nb={question}
        and chatgpt_highlevel.{column} is not null
        order by chatgpt_highlevel.{column} desc"""
        df = self.get_by_sql(sql)
        return df

    def get_highlevel_linked(self, question: int, empathy: str, positive: bool):
        column = empathy + ("2" if positive else "0")
        sql = f"""select chatgpt_highlevel.id,
                chatgpt_highlevel.index as main_topic,
                chatgpt_highlevel.{column}::float/100 as score, 
                topic0.id as topic0_id,
                coalesce(topic1.id,0) as topic1_id, topic1.label as topic1, 
                coalesce(topic2.id,0) as topic2_id, topic2.label as topic2,
                gpt_comment.comment as explaination
                from chatgpt_highlevel
                left join topic as topic0 on topic0.label=chatgpt_highlevel.index
                    and topic0.source='highlevel'
                left join topic as topic1 on topic1.label=split_part(chatgpt_highlevel.index,' ',1)
                    and topic1.source='gpt-4o-mini'
                left join topic as topic2 on topic2.label=split_part(chatgpt_highlevel.index,' ',2)
                    and topic2.source='gpt-4o-mini'
                left join gpt_comment on gpt_comment.topic_id=topic0.id
                    and gpt_comment.question_nb={question}
                    and gpt_comment.empathy='{empathy}'
                    and gpt_comment.positive is {str(positive).upper()}
                where chatgpt_highlevel.question_nb={question}
                and chatgpt_highlevel.{column} is not null
                order by chatgpt_highlevel.{column} desc"""
        df = self.get_by_sql(sql)
        return df

