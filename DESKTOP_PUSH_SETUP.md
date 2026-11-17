# Desktop push notification setup

MediMind web push uses a browser service worker, the Web Push protocol, and a scheduled GitHub Actions run. This lets reminders arrive after the MediMind tab is closed, provided the browser and operating system allow notifications.

## Required configuration

1. Generate a VAPID public/private key pair. Keep the private key secret.
2. Add these environment variables to the backend deployment:
   - `VAPID_PUBLIC_KEY`
   - `VAPID_PRIVATE_KEY`
   - `VAPID_SUBJECT` (a contact URI such as `mailto:owner@example.com`)
   - `REMINDER_CRON_SECRET` (a long random secret)
3. Add `EXPO_PUBLIC_VAPID_PUBLIC_KEY` to the frontend build environment, using the matching public key, then redeploy the website.
4. Add GitHub Actions repository secrets:
   - `MEDIMIND_BACKEND_URL` (the backend origin, without a trailing path)
   - `REMINDER_CRON_SECRET` (the same secret configured on the backend)
5. Merge the workflow in `.github/workflows/reminder-push.yml`. It calls the backend every five minutes and can also be run manually from GitHub Actions.
6. Sign in to MediMind on each desktop browser and choose **Enable desktop notifications** from Profile. Allow browser notifications when prompted.

Subscriptions are stored per account and are removed when the user disables push or signs out. The creator PDF evaluation option is shown only to the configured creator email and the backend independently checks that email before accepting or reading an evaluation.

The evaluation endpoint accepts PDFs up to 50 MB and 100 pages. It runs every page through the existing prescription extraction function, stores page-level outputs separately from normal prescription history, and reports extraction results. It does not calculate accuracy scores; those require matching human-verified answers for the same pages.
