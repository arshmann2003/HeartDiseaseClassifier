-- Reports for the usage events table. Paste any query into the Supabase SQL editor.
-- Keep-alive visits are logged as event = 'keepalive' and excluded from visitor counts.


-- Real visits per day, by where the visitor came from (?ref= tag on the link)
select
  date_trunc('day', created_at at time zone 'America/Vancouver')::date as day,
  ref,
  count(*) as visits
from events
where event = 'session_start'
group by 1, 2
order by 1 desc, visits desc;


-- All-time visits and engagement by source
select
  s.ref,
  count(distinct s.session_id) as visits,
  count(distinct p.session_id) as clicked_a_preset,
  count(distinct e.session_id) as entered_own_answers,
  round(100.0 * count(distinct coalesce(p.session_id, e.session_id)) / count(distinct s.session_id), 1)
    as pct_engaged
from events s
left join events p on p.session_id = s.session_id and p.event = 'preset_clicked'
left join events e on e.session_id = s.session_id and e.event = 'answers_edited'
where s.event = 'session_start'
group by s.ref
order by visits desc;


-- Which example profiles people try
select properties ->> 'preset' as preset, count(*) as clicks
from events
where event = 'preset_clicked'
group by 1
order by clicks desc;


-- Most recent real visits
select created_at at time zone 'America/Vancouver' as visited_at, ref
from events
where event = 'session_start'
order by created_at desc
limit 25;


-- Keep-alive health: when the scheduled visit last reached the app
select max(created_at at time zone 'America/Vancouver') as last_keepalive, count(*) as total_keepalives
from events
where event = 'keepalive';
