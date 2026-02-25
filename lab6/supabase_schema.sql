-- Supabase schema for the preference labeling app
-- Run this in your Supabase SQL editor to set up the table.

create table if not exists preferences (
    id                  uuid primary key default gen_random_uuid(),

    -- Core RLHF fields
    prompt              text not null,
    chosen              text,               -- null when is_tie = true
    rejected            text,               -- null when is_tie = true
    preference          text not null,      -- 'A' | 'B' | 'Tie'
    is_tie              boolean not null default false,

    -- Response A
    response_a          text not null,
    response_a_model    text,
    response_a_temp     float,
    response_a_tokens   int,
    response_a_latency  float,

    -- Response B
    response_b          text not null,
    response_b_model    text,
    response_b_temp     float,
    response_b_tokens   int,
    response_b_latency  float,

    -- Prompt metadata
    prompt_id           int,
    prompt_tension      text,
    prompt_difficulty   text,

    -- Session metadata
    session_id          text,
    labeled_at          timestamptz default now()
);

-- Index for downstream queries
create index if not exists preferences_session_idx on preferences(session_id);
create index if not exists preferences_prompt_idx on preferences(prompt_id);
create index if not exists preferences_is_tie_idx on preferences(is_tie);

-- Enable Row Level Security (for Supabase anon key usage)
alter table preferences enable row level security;

-- Allow inserts from anon key (labeling app)
create policy "Allow inserts" on preferences
    for insert to anon with check (true);

-- Allow reads (for export queries)
create policy "Allow reads" on preferences
    for select to anon using (true);
