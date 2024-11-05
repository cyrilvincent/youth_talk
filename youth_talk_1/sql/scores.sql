select topic.id, topic.label, count(form_topic.id) as nb_neg_form, sub_topic.nb_form, power(count(form_topic.id)::float,2)/sub_topic.nb_form as ratio from topic
join form_topic on form_topic.topic_id=topic.id
join form on form_topic.form_id=form.id
join stat on stat.id=form.id
join (select topic.id, count(form_topic.id) as nb_form from topic
	join form_topic on form_topic.topic_id=topic.id
	join form on form_topic.form_id=form.id
	join stat on stat.id=form.id
	where source='textrank'
	group by topic.id
	having count(form_topic.id) > 10
) sub_topic on sub_topic.id = topic.id
where source='textrank'
and stat.empathy_category=0
group by topic.id, sub_topic.nb_form
having count(form_topic.id) > 1
order by ratio desc

