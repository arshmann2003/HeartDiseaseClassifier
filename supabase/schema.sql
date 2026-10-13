-- Usage events for the Heart Disease Risk Screener web app.
-- Run once in the Supabase SQL editor.

create table if not exists public.events (
  id          bigint generated always as identity primary key,
  created_at  timestamptz not null default now(),
  session_id  uuid not null,
  event       text not null,
  ref         text not null default 'direct' check (char_length(ref) <= 40),
  properties  jsonb not null default '{}'::jsonb
);

create index if not exists events_created_at_idx on public.events (created_at);
create index if not exists events_session_idx on public.events (session_id);

-- The app ships a public key, so lock the table down: that key may insert
-- known event types only, and cannot read, update or delete anything.
alter table public.events enable row level security;

grant insert on public.events to anon;

drop policy if exists "app can insert events" on public.events;
create policy "app can insert events"
  on public.events
  for insert
  to anon
  with check (
    event in ('session_start', 'keepalive', 'preset_clicked', 'answers_edited')
    and pg_column_size(properties) < 1024
  );
