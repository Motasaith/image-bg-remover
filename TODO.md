# TODO: Clean up and optimize main.py

- [ ] Remove unused import: `import shutil`
- [ ] Remove the entire Job Queue System (process_video_job, create_job, get_job_status)
- [ ] Replace hardcoded FFmpeg parameters in `encode_cmd` with environment variables (`FFMPEG_CODEC`, `FFMPEG_CRF`, `FFMPEG_BV`)
- [ ] Replace hardcoded cleanup intervals (3600 and 86400) with `CLEANUP_INTERVAL_SECONDS` and `FILE_EXPIRY_SECONDS`
- [ ] Replace all hardcoded "processed" directory references with `PROCESSED_DIR`
- [ ] Add graceful shutdown handling with signal handlers
- [ ] Add model warm-up in the startup event to load RMBG and RVM models once
- [ ] Update the static files mount to use `PROCESSED_DIR`
- [ ] Update home route to remove /jobs reference
