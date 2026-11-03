// Copy to web/config.js for a local preview that records anonymous usage.
// Use the Supabase publishable key only. That key can insert known event
// names and cannot read the table. GitHub Actions can inject the same values
// from SUPABASE_URL and SUPABASE_KEY at deploy time instead of committing them.
window.ANALYTICS = {
  url: "https://YOUR_PROJECT.supabase.co",
  key: "YOUR_PUBLISHABLE_KEY",
};
