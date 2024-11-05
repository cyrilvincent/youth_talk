select topic.*, count(form_topic.id) as nb_form from topic
join form_topic on form_topic.topic_id=topic.id
where source='textrank'
group by topic.id
order by nb_form desc

select topic.*, lema.label, lema.count from topic
join lema on lema.topic_id=topic.id
where topic.label = 'war'
and source='textrank'
order by lema.count desc