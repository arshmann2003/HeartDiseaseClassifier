-- Read-only views over the events table. Run once in the Supabase SQL editor after schema.sql.
-- Re-running is safe. Keep-alive bot visits are excluded.
--
-- security_invoker makes each view obey the events table's row-level security, so the app's
-- public key still cannot read anything through them. You can read them from the dashboard.


-- One row per real visit, with what the visitor did
create or replace view public.visits
with (security_invoker = true) as
select
  s.created_at at time zone 'America/Vancouver' as visited_at,
  s.ref,
  exists (
    select 1 from public.events p where p.session_id = s.session_id and p.event = 'preset_clicked'
  ) as clicked_preset,
  exists (
    select 1 from public.events e where e.session_id = s.session_id and e.event = 'answers_edited'
  ) as edited_answers,
  s.session_id
from public.events s
where s.event = 'session_start'
order by s.created_at desc;


-- Visit counts per day and source
create or replace view public.daily_visits
with (security_invoker = true) as
select
  (created_at at time zone 'America/Vancouver')::date as day,
  ref,
  count(*) as visits
from public.events
where event = 'session_start'
group by 1, 2
order by 1 desc, visits desc;


revoke all on public.visits, public.daily_visits from anon, authenticated;
