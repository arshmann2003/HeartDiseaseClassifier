-- Read-only database login for the local analytics dashboard (dashboard.py).
-- Run once in the Supabase SQL editor after schema.sql and views.sql.
--
-- Replace the password IN THE SQL EDITOR, not in this file, so it never gets committed.
-- Use letters and numbers only (20+ characters) to avoid escaping problems.

do $$
begin
  if exists (select from pg_roles where rolname = 'analytics_reader') then
    alter role analytics_reader with login password 'REPLACE_WITH_A_STRONG_PASSWORD';
  else
    create role analytics_reader with login password 'REPLACE_WITH_A_STRONG_PASSWORD';
  end if;
end $$;

-- Every session for this login is read-only, even if a query tries to write.
alter role analytics_reader set default_transaction_read_only = on;

grant usage on schema public to analytics_reader;
grant select on public.events, public.visits, public.daily_visits to analytics_reader;

-- Row-level security is on for events, so the reader needs its own select policy.
drop policy if exists "dashboard can read events" on public.events;
create policy "dashboard can read events"
  on public.events
  for select
  to analytics_reader
  using (true);
