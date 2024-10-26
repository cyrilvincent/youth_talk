select * from form
where question_01_contrib1_answer is not null
and length(question_01_contrib1_answer) > 5
and empathy_answers > 0

select * from form
where question_01_contrib1_answer is not null
and length(question_01_contrib1_answer) + COALESCE (length(question_01_contrib2_answer), 0) + COALESCE(length(question_01_contrib3_answer),0) > 40
and empathy_answers > 0

select empathy_pd_6, empathy_pd_17, empathy_pd_24, empathy_pd_27, (COALESCE(empathy_pd_6,3) + COALESCE(empathy_pd_17,3) + COALESCE(empathy_pd_24,3) + COALESCE(empathy_pd_27,3)) / 4 from form
where empathy_answers > 0

Prendre PD
1- 2.66 Low
3.34-5 High