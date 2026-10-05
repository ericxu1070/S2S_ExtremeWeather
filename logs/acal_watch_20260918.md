# acal watch — array 1217 (+1222 rerun), 2026-09-18

## Handover state (watch started 04:08 UTC)
- 1217_0 e02_c4_20210218 job 1218 COMPLETED 04:05:31 — compared+pruned (prior operator)
- 1217_1 e12_c3_20221121 job 1219 COMPLETED 03:43:43 — compared+pruned (prior operator)
- 1217_2 e13_c3_20221225 job 1220 COMPLETED 03:59:04 — compared+pruned (prior operator)
- 1217_3 e14_h4_20230104 job 1221 FAILED 02:07:03 — foreign tenant on node 6 -> CUDA OOM leg-3 score; resubmitted as 1222_3 (PENDING, pinned node 0)
- 1217_4 e23_h3_20231228 job 1223 RUNNING node 0 — leg-2 score in flight
- 1217_5 e24_c3_20240119 job 1224 RUNNING node 4 — leg 1
- 1217_6 e25_h3_20240131 job 1225 RUNNING node 2 — leg 1
- 1217_[7-11] PENDING (throttle 3), pinned node 0
- df -h /home at 04:08 UTC: 385G avail (97% used)

## 04:08 UTC preflight checks on the three live tasks
- 1223 e23_h3: READY yes, N=32, tail HIGH (correct, _h), .err empty
- 1224 e24_c3: READY yes, N=32, tail LOW (correct, _c), .err empty
- 1225 e25_h3: READY yes, N=32, tail HIGH (correct, _h), .err empty

## Node survey 04:09 UTC — NO free node
- node0 ours (1223) 67.5GB x8 | node2 ours (1225) | node4 ours (1224) 80.6GB x8
- node1 FOREIGN 62-66GB x8 | node3 FOREIGN 35-80GB x8 | node5 FOREIGN 31-78GB x8
- node6 FOREIGN 63-67GB x8 (the 01:51 tenant, still there) | node7 FOREIGN 1-74GB x8
- All 3 running slots in use; throttle stays 3, no widening possible.

## Verified prior-operator results (read-only audit, 04:12 UTC)
- e02_c4_20210218 job1218 node6 wall 245.3min logZ=-1.1815 norm=0.496 founders=5 unw=-5.753 wtd=-3.273
    ESS 32.0/15.9/8.3/16.6/22.2  killed 0/11/16/11/7  maxmult 1/5/8/5/2
- e12_c3_20221121 job1219 node0 wall 223.4min logZ=-0.8544 norm=1.269 founders=11 unw=-1.879 wtd=+0.382
    ESS 32.0/15.1/14.5/14.4/22.0  killed 0/12/13/10/8  maxmult 1/5/4/5/4
- e13_c3_20221225 job1220 node0 wall 238.8min logZ=-0.7580 norm=1.018 founders=7 unw=-3.556 wtd=-1.127
    ESS 32.0/14.9/16.2/11.5/6.8   killed 0/11/12/16/17  maxmult 1/4/4/6/10
- all three pruned (0 state.nc left, 9.1G footprint each); .err empty; 0 *.nc.tmp.* repo-wide
- norm check range 0.50-1.27: inside run_aires' 0.2..5.0 no-warning band and consistent with
  the documented open problem 5 (aplots.py:179); not treated as a defect.

## INCIDENT 1 (pre-handover, confirmed by evidence) — job 1221 e14_h4_20230104
- leg-3 FCN3 score, all 8 shards exited 1 on all 3 attempts. Shard logs: 192x "CUDA out of memory";
  foreign PID 484928 holding 61.7-65.7 GiB on GPU 0 of 79.18 GiB, leaving ~2.2 GiB.
  Our shard needs ~13-15 GiB. Classification: foreign-tenant contention on node 6, NOT a code fault.
- Resubmitted as 1222_3, pinned to node 0. Legs 1-2 + leg-3 walker states banked -> resumes cheaply.

## 04:12 UTC survey — no free node; all healthy
- node0 OURS(1223) 67541x8 | node2 OURS(1225) spinning up | node4 OURS(1224) spinning up
- node1/3/5/6/7 FOREIGN (unchanged). df /home 375G avail. 1223 leg-2 score, 1224+1225 leg-1 walk.
- 04:20 survey: no free node (1/3/5/6/7 still FOREIGN, unchanged). 1223 leg-2 score,
  1224 leg-2 walk (leg1 pool 23.8 min), 1225 leg-1 walk. All .err empty.
  df /home 370G (385->375->370 over 12 min; walk stage self-prunes segments <= n-2, expect plateau).

## INCIDENT 2 — 04:04:36 UTC, job 1225 e25_h3_20240131, node 2, leg-1 walk, shards 4/5/6
- Symptom: 3 of 8 walk shards exited 1 ~1 min after start. try1 shard4/5/6:
  arena claim logged (limit=71.26 GiB) but the E-line probe cascade shows the
  preallocation actually failed all the way down (71.26 -> 64.13 -> ... -> 16.30 GiB);
  model still loaded (15 s), then death on a TINY alloc --
  walker.py:359 segment_key -> jax.random.PRNGKey -> RESOURCE_EXHAUSTED CUDA_ERROR_OUT_OF_MEMORY.
  Note: the same E-line cascade appears in the SURVIVING shards (shard0 peak 21.17-21.31 GiB,
  limit 71.26) -- the cascade is allocator back-off noise, not the fault signature.
- Attribution: GPUs 4/5/6 of node 2 had no free memory at 04:04:36 but read 0 MB at 04:12.
  No job of ours was on node 2 before (1217_3 ended 02:14 on node 6; 1217_6 started 04:03:43),
  so the holder was a SHORT-LIVED FOREIGN process that released by 04:12.
  => routine foreign-tenant contention, the documented a3mega mode. NOT a code fault, NOT critical.
- Response: none required; the built-in retry handled it. try2 (04:29) has all 8 shards claimed;
  shards 0-3,7 exited instantly on cache hits, shards 4/5/6 now rolling at 73533 MB on node 2.
  Nothing killed, no foreign process touched.
- Cost: ~25 min of leg-1 wall time on this case.
- 04:38 survey: no free node. 1223 score-leg02 (~31 min in, expect ~38), 1224 score-leg02 started,
  1225 leg-1 walk try2 rolling (node2 GPUs 4/5/6 at 80629). node4 reads all-zero = between pools,
  NOT free. df 349G. 1225.err 480B = the 3 try1 shard lines from incident 2 (expected).

## 04:46 UTC — NODE 3 FREED, given to the e14 rerun
- 04:45 survey: node3 went from FOREIGN (35193-80851 x8 at 04:38) to 0 MB x8.
  Verified free: nvidia-smi --query-compute-apps EMPTY, sinfo idle, not one of our jobs.
  (node1 remains sinfo-idle but FOREIGN 62-66 GB x8 -- the documented trap, correctly skipped.)
- df /home 359G (> 100G floor; rose from 349G as the walk stage self-pruned old segments).
- Action: scontrol update JobId=1222_3 ExcNodeList=nucla3m-a3meganodeset-[1,2,4,5,6,7]
  (was [1-7]). ONE task widened only. 1217_[7-11] left excluding node 3 on purpose so they
  cannot cascade onto it. 1222 sits outside the 1217 throttle, so no throttle change made.
- Result: 1222_3 RUNNING on nucla3m-a3meganodeset-3 within seconds. log = logs/Vayuh-s2s-1222.out
- 1222_3 startup checks PASS: READY yes, N=32 (not 64), tail HIGH (correct for _h).
  Resume verified: cached 96/224 walker segments + 32/128 score cubes, 64 states resident;
  legs 1-2 replayed from cache (DMC step 2 ESS=17.1/32 killed=9), leg-3 walk verified complete,
  now re-running the leg-3 FCN3 score that OOM'd on node 6. 4 cases now in flight.
- 04:54 survey: 4 in flight, all clean. node0(1223) walk-leg03, node2(1225) walk-leg02
  (leg1 pool 44.2 min = 20 nominal + incident-2 retry), node3(1222) score-leg03 67541MB/card
  with no competitor, node4(1224) score-leg02. Foreign: 1,5,6,7. No free node. df 346G.

## 05:03 UTC — CONTEXT CHANGE: the cluster is no longer ours alone
- node5 read 0 MB x8 at 05:02 and looked free. VERIFIED BEFORE ACTING: squeue shows job 1241
  `dsr1_eval_n7` (user ubuntu, NOT our campaign -- our jobs are all named Vayuh-s2s) started on
  node5 13 s earlier. The zeros were ITS startup. NOT free -> no widening done. Nothing touched.
- New competition now queued that was absent at 04:08:
    gm_withnuc: 1228 c3_slot (ReqNodeNotAvail), 1227 b6_p31_clock (Resources)
    ubuntu (other project): 1241 dsr1_eval_n7 RUNNING node5; 1242/1243/1244/1245 dsr1_eval_*
      (Dependency); 1234 d5_converge, 1235 d5_dfhalf (Resources)
- node0 also read all-zero at 05:02 but that is OURS (1223) between walk-leg03 and score-leg03.
- Standing risk for whoever holds this next: 1217_[7-11] are pinned to node 0 ALONE. With a
  contended queue that is now a throughput bottleneck, not just a safety measure. Widen them
  one at a time as genuinely-free nodes appear, but expect to lose races to the other jobs.
- df 325G. 4 of our cases in flight (1222_3 node3, 1223 node0, 1224 node4, 1225 node2).
- 05:10 survey: all 4 of ours in FCN3 score pools (1222 leg3, 1223 leg3, 1224 leg2, 1225 leg2).
  node2 reads 4MB/card and node4 reads 0 on 6 of 8 -- that is the documented ~9 min concurrent
  NFS model-pickle load, not idleness. No free node (node5 = teammate 1241). df 322G.
  Disk drift ~-1 GB/min net since 04:08 (385->322), with a rise at 04:45 when a walk self-pruned.
  ~240G of headroom over the 80G floor; first prune (1223, ETA ~07:30) should return ~45G.

## INCIDENT 3 — ~04:39 UTC, job 1224 e24_c3_20240119, node 4, leg-2 FCN3 score, shards 0-5
- Symptom: 6 of 8 score shards exited 1, each failing EXACTLY ONE task (w00..w05 @6d) at model
  load (torch module.to(), Tried to allocate 718 MiB). Shards 6,7 completed all 4 tasks.
  The 6 shards then went on to finish their OTHER 3 tasks each, so only 6 tasks need redoing.
- ATTRIBUTION (decisive): all six shards report their LOCAL "GPU 0" -- each is pinned to a
  different PHYSICAL card by CUDA_VISIBLE_DEVICES -- yet all six cite the SAME holder,
  "Process 4161162 has 78.50 GiB". One process resident on six different physical GPUs.
  Our acal shards are strictly single-GPU, and 78.50 GiB is not our FCN3 size (~66 GiB / 67541 MB).
  => a transient multi-GPU FOREIGN process took node4 GPUs 0-5 at ~04:39 and released by 04:45
  (node4 read 67541 x8 at 04:45/04:54/05:02). PID 4161162 confirmed GONE at 05:19, no compute
  apps on node4. Same class as incidents 1 and 2: foreign-tenant contention, NOT a code fault.
- Response: none needed. try2 launched 05:17, all 6 shards loading clean (oom=0). Nothing killed.
- PATTERN WORTH ESCALATING: three foreign-tenant collisions in ~3.5 h -- node6 01:51 (persistent,
  killed job 1221), node2 GPUs 4-6 04:04 (transient), node4 GPUs 0-5 04:39 (transient).
  The a3mega nodes are heavily contended by non-Slurm tenants tonight; sinfo idle means nothing.
- 05:19 state: df 334G (recovered from 322G via self-prune). 1222 walk-leg04 node3,
  1223 score-leg03 node0, 1224 score-leg02 try2 node4, 1225 score-leg02 node2. No free node.
- 05:27 survey: all clean. node0(1223) score-leg03, node2(1225) score-leg02, node3(1222)
  walk-leg04 80629x8, node4(1224) score-leg02 try2 67541 on GPUs0-5 (6,7 idle = already cached,
  correct). No free node. df 320G.
- 05:36: 1222 score-leg04 (walk-leg04 19.7min), 1223 walk-leg04 (score-leg03 29.3min),
  1224 walk-leg03 (score-leg02 51.7min incl. incident-3 retry), 1225 score-leg02. df 349G.
  node3 zeros = ours between pools. No free node; competing queue unchanged.
- 05:44: 1222 score-leg04, 1223 walk-leg04, 1224 walk-leg03, 1225 score-leg02 (34min in). df 335G. No free node. errs = incident lines only (1224 972B, 1225 480B).
- 05:53: 1222 score-leg04, 1223 score-leg04 (walk4 19.8min), 1224 score-leg03 (walk3 23.6min), 1225 walk-leg03 (score2 38.7min). df 337G. No free node.
- 06:01: 1222 walk-leg05 (score4 23.5min), 1223 score-leg04, 1224 score-leg03, 1225 walk-leg03. df 337G stable. No free node.
- 06:09: 1222 walk-leg05, 1223 score-leg04, 1224 score-leg03, 1225 score-leg03 (walk3 19.7min). df 335G. No free node.
- 06:17: 1222 walk-leg05, 1223 walk-leg05 (score4 23.1min), 1224 score-leg03, 1225 score-leg03. df 340G. No free node.
- 06:25: 1222 score-leg05 (walk5 19.6min), 1223 walk-leg05, 1224 walk-leg04 (score3 29.4min), 1225 score-leg03. df 341G. No free node.
- 06:33: 1222 score-leg05, 1223 walk-leg05 (node0 GPUs 3/5/6 idle = shards done early, .err still 0), 1224 walk-leg04, 1225 score-leg03. df 330G. No free node.
- 06:42: 1222 walk-leg06 CARRY (score5 16.7min), 1223 score-leg05 (walk5 19.7min), 1224 walk-leg04, 1225 walk-leg04 (score3 29.5min). df 352G. errs unchanged. No free node.

## 2026-09-18T06:49Z watchdog (slurm/acal_watchdog.sh, pid 3080662) took over the watch; it writes DONE/CRITICAL to logs/acal_trigger_20260918.txt
- 2026-09-18T06:49Z 1217_7 ExcNodeList -> foreign set [1 5 6 7]; free [none]; df 336G
- 2026-09-18T06:49Z 1217_8 ExcNodeList -> foreign set [1 5 6 7]; free [none]; df 336G
- 2026-09-18T06:49Z 1217_9 ExcNodeList -> foreign set [1 5 6 7]; free [none]; df 336G
- 2026-09-18T06:49Z 1217_10 ExcNodeList -> foreign set [1 5 6 7]; free [none]; df 336G
- 2026-09-18T06:49Z 1217_11 ExcNodeList -> foreign set [1 5 6 7]; free [none]; df 336G
- 2026-09-18T06:49Z watchdog: running 1222_3@n3 1217_6@n2 1217_5@n4 1217_4@n0 ; pending 1217_7 1217_8 1217_9 1217_10 1217_11 ; free [none]; foreign [1 5 6 7]; df 336G
- 2026-09-18T07:09Z throttle 1217 3 -> 4 (running 3, free nodes [3], df 302G)

## 2026-09-18T07:11Z case 3 e14_h4_20230104 landed (watchdog): compare + prune done, df 313G -> 313G
- job 1222 on nucla3m-a3meganodeset-3, wall 142.2 min; log_Z -0.885, founders 8, normalization_check 0.318, ESS by step [32.0,17.1,20.0,20.5,19.4]
- 2026-09-18T07:16Z 1217_11 ExcNodeList -> foreign set [1 3 5 6 7]; free [none]; df 349G
- 2026-09-18T07:16Z 1217_10 ExcNodeList -> foreign set [1 3 5 6 7]; free [none]; df 349G
- 2026-09-18T07:16Z 1217_9 ExcNodeList -> foreign set [1 3 5 6 7]; free [none]; df 349G
- 2026-09-18T07:16Z 1217_8 ExcNodeList -> foreign set [1 3 5 6 7]; free [none]; df 349G
- 2026-09-18T07:16Z 1217_7 ExcNodeList -> foreign set [1 3 5 6 7]; free [none]; df 349G
- 2026-09-18T07:16Z throttle 1217 4 -> 3 (running 3, free nodes [none], df 349G)

## 2026-09-18T07:28Z case 4 e23_h3_20231228 landed (watchdog): compare + prune done, df 348G -> 348G
- job 1223 on nucla3m-a3meganodeset-0, wall 218.6 min; log_Z -1.545, founders 10, normalization_check 3.161, ESS by step [32.0,17.2,11.0,14.5,27.1]

## 2026-09-18T07:31Z watchdog (slurm/acal_watchdog.sh, pid 3085840) took over the watch; it writes DONE/CRITICAL to logs/acal_trigger_20260918.txt
- 2026-09-18T07:31Z watchdog: running 1217_7@n0 1217_6@n2 1217_5@n4 ; pending 1217_11 1217_10 1217_9 1217_8 ; free [none]; foreign [1 3 5 6 7]; df 382G
- 2026-09-18T07:33Z acal_ctl (ubuntu): held pending 1217_11 (case 11 e42_h4_20251228)
- 2026-09-18T07:33Z acal_ctl (ubuntu): held pending 1217_10 (case 10 e37_h3_20250302)
- 2026-09-18T07:33Z acal_ctl (ubuntu): held pending 1217_9 (case 9 e36_c3_20250223)
- 2026-09-18T07:33Z acal_ctl (ubuntu): held pending 1217_8 (case 8 e34_c4_20250125)
- 2026-09-18T07:33Z acal_ctl (ubuntu): PAUSED (drain): 7 task(s) handled; resume with: bash slurm/acal_ctl.sh resume

## 2026-09-18T07:38Z watchdog (slurm/acal_watchdog.sh, pid 3087486) took over the watch; it writes DONE/CRITICAL to logs/acal_trigger_20260918.txt
- 2026-09-18T08:04Z acal_ctl (ubuntu): released 1217_11 (case 41 e42_h4_20251228)
- 2026-09-18T08:04Z acal_ctl (ubuntu): released 1217_10 (case 36 e37_h3_20250302)
- 2026-09-18T08:04Z acal_ctl (ubuntu): released 1217_9 (case 35 e36_c3_20250223)
- 2026-09-18T08:04Z acal_ctl (ubuntu): released 1217_8 (case 33 e34_c4_20250125)
- 2026-09-18T08:04Z acal_ctl (ubuntu): RESUMED: submitted job 1250 (rung 2) --array=0,2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,25,26,27,28,29,30,31,34,37,38,39,40%1 excluding [1 3 5 6 7]

## 2026-09-18T08:04Z watchdog (slurm/acal_watchdog.sh, pid 3092413) took over the watch; it writes DONE/CRITICAL to logs/acal_trigger_20260918.txt
- 2026-09-18T08:04Z watchdog: running 1217_7@n0 1217_6@n2 1217_5@n4 ; pending 1250_0 1250_2 1250_3 1250_4 1250_5 1250_6 1250_7 1250_8 1250_9 1250_10 1250_14 1250_15 1250_16 1250_17 1250_18 1250_19 1250_20 1250_21 1250_25 1250_26 1250_27 1250_28 1250_29 1250_30 1250_31 1250_34 1250_37 1250_38 1250_39 1250_40 1217_11 1217_10 1217_9 1217_8 ; free [none]; foreign [1 3 5 6 7]; df 857G

## 2026-09-18T08:04Z the 42-case slate opened (ACAL_RUNG=2 resume); old rung-3 watchdog stopped
- HRRR 2015/2016/2017/2022 deleted 08:03:58-08:04:31Z (5,802 files, 497 GiB; manifest
  downscaler/docs/hrrr_deleted_2026-09-18.txt): df 362G -> 791G -> 853G.
- controller + watchdog made rung-aware (slurm/acal_slate.sh; rung_<array> records:
  1217=3, 1222=3, 1250=2). Verified in DRY: 1217_5/6/7 -> rung-2 k=23/24/32, held 8-11 -> 33/35/36/41.
- released 1217_8..11 (re-pinned to exclude 1,3,5,6,7); submitted array 1250 = the 30
  rung-2-only cases, throttle 1 (no free node; foreign 1,3,5,6,7). Watchdog pid 3092413, slate=42.
- campaign rung remembered in logs/acal_watchdog_state/rung (=2); bare `acal_ctl.sh status` shows 42.
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_0 (case 0 e01_h2_20210119)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_2 (case 2 e03_h2_20210411)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_3 (case 3 e04_h2_20210610)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_4 (case 4 e05_h2_20211011)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_5 (case 5 e06_h2_20211206)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_6 (case 6 e07_h2_20211217)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_7 (case 7 e08_h2_20211229)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_8 (case 8 e09_c2_20220228)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_9 (case 9 e10_c2_20220314)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_10 (case 10 e11_h2_20220321)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_14 (case 14 e15_h2_20230118)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_15 (case 15 e16_c2_20230205)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_16 (case 16 e17_c2_20230322)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_17 (case 17 e18_h2_20231006)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_18 (case 18 e19_h2_20231025)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_19 (case 19 e20_c2_20231103)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_20 (case 20 e21_h2_20231120)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_21 (case 21 e22_h2_20231210)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_25 (case 25 e26_h2_20240211)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_26 (case 26 e27_h2_20240228)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_27 (case 27 e28_h2_20240318)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_28 (case 28 e29_h2_20240418)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_29 (case 29 e30_h2_20241001)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_30 (case 30 e31_h2_20241027)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_31 (case 31 e32_h2_20241221)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_34 (case 34 e35_h2_20250209)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_37 (case 37 e38_h2_20250317)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_38 (case 38 e39_h2_20250331)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_39 (case 39 e40_h2_20250930)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1250_40 (case 40 e41_h2_20251120)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1217_11 (case 41 e42_h4_20251228)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1217_10 (case 36 e37_h3_20250302)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1217_9 (case 35 e36_c3_20250223)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled pending 1217_8 (case 33 e34_c4_20250125)
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled RUNNING 1217_7 on 0 (case 32 e33_h4_20250101, was in score-leg02); it resumes from its last finished leg
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled RUNNING 1217_6 on 2 (case 24 e25_h3_20240131, was in walk-leg06); it resumes from its last finished leg
- 2026-09-18T08:09Z acal_ctl (ubuntu): cancelled RUNNING 1217_5 on 4 (case 23 e24_c3_20240119, was in walk-leg06); it resumes from its last finished leg
- 2026-09-18T08:09Z acal_ctl (ubuntu): PAUSED: 37 task(s) handled; resume with: bash slurm/acal_ctl.sh resume
- 2026-09-29T16:37Z acal_ctl (ubuntu): RESUMED: submitted job 2020 (rung 2) --array=0,2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41%5 excluding [4 5 7]
- 2026-09-29T17:33Z watchdog: running 2020_2@n6 2020_3@n1 2020_4@n2 2020_5@n0 2020_0@n3 ; pending 2020_6 2020_7 2020_8 2020_9 2020_10 2020_14 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 666G
- 2026-09-29T18:34Z watchdog: running 2020_2@n6 2020_3@n1 2020_4@n2 2020_5@n0 2020_0@n3 ; pending 2020_6 2020_7 2020_8 2020_9 2020_10 2020_14 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 771G
- 2026-09-29T19:35Z watchdog: running 2020_2@n6 2020_3@n1 2020_4@n2 2020_5@n0 2020_0@n3 ; pending 2020_6 2020_7 2020_8 2020_9 2020_10 2020_14 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 767G
- 2026-09-29T20:36Z watchdog: running 2020_2@n6 2020_3@n1 2020_4@n2 2020_5@n0 2020_0@n3 ; pending 2020_6 2020_7 2020_8 2020_9 2020_10 2020_14 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 728G

## 2026-09-29T20:48Z case 3 e04_h2_20210610 landed (watchdog): compare + prune done, df 708G -> 708G
- job 2023 on nucla3m-a3meganodeset-1, wall 245.2 min; log_Z -0.682, founders 7, normalization_check 0.777, ESS by step [32.0,11.0,2.5,10.4,17.0]

## 2026-09-29T21:00Z case 4 e05_h2_20211011 landed (watchdog): compare + prune done, df 717G -> 717G
- job 2024 on nucla3m-a3meganodeset-2, wall 259.4 min; log_Z -1.120, founders 8, normalization_check 2.293, ESS by step [32.0,16.8,15.6,9.0,21.1]

## 2026-09-29T21:27Z case 2 e03_h2_20210411 landed (watchdog): compare + prune done, df 705G -> 705G
- job 2022 on nucla3m-a3meganodeset-6, wall 286.5 min; log_Z -0.661, founders 8, normalization_check 0.358, ESS by step [32.0,16.4,14.0,10.7,12.3]
- 2026-09-29T21:42Z watchdog: running 2020_8@n6 2020_7@n2 2020_6@n1 2020_5@n0 2020_0@n3 ; pending 2020_9 2020_10 2020_14 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 683G

## 2026-09-29T21:49Z case 0 e01_h2_20210119 landed (watchdog): compare + prune done, df 669G -> 669G
- job 2021 on nucla3m-a3meganodeset-3, wall 308.9 min; log_Z -0.384, founders 9, normalization_check 1.480, ESS by step [32.0,20.2,12.3,9.2,12.4]

## 2026-09-29T21:51Z case 5 e06_h2_20211206 landed (watchdog): compare + prune done, df 706G -> 706G
- job 2025 on nucla3m-a3meganodeset-0, wall 310.6 min; log_Z -1.436, founders 5, normalization_check 0.305, ESS by step [32.0,12.2,15.7,10.8,19.9]
- 2026-09-29T22:47Z watchdog: running 2020_10@n0 2020_9@n3 2020_8@n6 2020_7@n2 2020_6@n1 ; pending 2020_14 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 664G
- 2026-09-29T23:48Z watchdog: running 2020_10@n0 2020_9@n3 2020_8@n6 2020_7@n2 2020_6@n1 ; pending 2020_14 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 702G

## 2026-09-30T00:46Z case 6 e07_h2_20211217 landed (watchdog): compare + prune done, df 666G -> 666G
- job 2027 on nucla3m-a3meganodeset-1, wall 240.7 min; log_Z -0.874, founders 7, normalization_check 0.179, ESS by step [32.0,17.4,19.4,7.4,10.0]
- 2026-09-30T00:51Z watchdog: running 2020_14@n1 2020_10@n0 2020_9@n3 2020_8@n6 2020_7@n2 ; pending 2020_15 2020_16 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 692G

## 2026-09-30T01:08Z case 7 e08_h2_20211229 landed (watchdog): compare + prune done, df 689G -> 689G
- job 2028 on nucla3m-a3meganodeset-2, wall 249.6 min; log_Z -0.602, founders 9, normalization_check 0.533, ESS by step [32.0,18.1,20.8,17.1,15.8]

## 2026-09-30T01:51Z case 8 e09_c2_20220228 landed (watchdog): compare + prune done, df 643G -> 643G
- job 2029 on nucla3m-a3meganodeset-6, wall 260.4 min; log_Z -0.734, founders 8, normalization_check 1.613, ESS by step [32.0,12.5,10.2,16.9,15.3]
- 2026-09-30T01:56Z watchdog: running 2020_16@n6 2020_15@n2 2020_14@n1 2020_10@n0 2020_9@n3 ; pending 2020_17 2020_18 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 696G

## 2026-09-30T02:28Z case 10 e11_h2_20220321 landed (watchdog): compare + prune done, df 632G -> 632G
- job 2058 on nucla3m-a3meganodeset-0, wall 276.1 min; log_Z -0.847, founders 8, normalization_check 1.347, ESS by step [32.0,12.4,13.6,17.7,10.9]

## 2026-09-30T02:35Z case 9 e10_c2_20220314 landed (watchdog): compare + prune done, df 661G -> 661G
- job 2057 on nucla3m-a3meganodeset-3, wall 286.3 min; log_Z -1.769, founders 4, normalization_check 0.603, ESS by step [32.0,19.6,13.4,4.6,14.9]
- 2026-09-30T03:01Z watchdog: running 2020_18@n3 2020_17@n0 2020_16@n6 2020_15@n2 2020_14@n1 ; pending 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 670G
- 2026-09-30T04:02Z watchdog: running 2020_18@n3 2020_17@n0 2020_16@n6 2020_15@n2 2020_14@n1 ; pending 2020_19 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 660G

## 2026-09-30T04:45Z case 14 e15_h2_20230118 landed (watchdog): compare + prune done, df 638G -> 638G
- job 2059 on nucla3m-a3meganodeset-1, wall 239.2 min; log_Z -0.802, founders 9, normalization_check 1.099, ESS by step [32.0,15.8,14.5,16.2,16.8]
- 2026-09-30T05:05Z watchdog: running 2020_19@n1 2020_18@n3 2020_17@n0 2020_16@n6 2020_15@n2 ; pending 2020_20 2020_21 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 652G

## 2026-09-30T05:12Z case 15 e16_c2_20230205 landed (watchdog): compare + prune done, df 631G -> 631G
- job 2060 on nucla3m-a3meganodeset-2, wall 243.9 min; log_Z -1.367, founders 9, normalization_check 0.570, ESS by step [32.0,12.0,13.0,19.3,19.4]

## 2026-09-30T06:05Z case 16 e17_c2_20230322 landed (watchdog): compare + prune done, df 604G -> 604G
- job 2064 on nucla3m-a3meganodeset-6, wall 258.7 min; log_Z -1.197, founders 9, normalization_check 1.753, ESS by step [32.0,16.7,12.9,15.4,24.3]
- 2026-09-30T06:10Z watchdog: running 2020_21@n6 2020_20@n2 2020_19@n1 2020_18@n3 2020_17@n0 ; pending 2020_23 2020_24 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 649G

## 2026-09-30T06:58Z case 17 e18_h2_20231006 landed (watchdog): compare + prune done, df 593G -> 593G
- job 2065 on nucla3m-a3meganodeset-0, wall 270.4 min; log_Z -0.939, founders 8, normalization_check 0.358, ESS by step [32.0,6.3,9.2,16.0,8.5]

## 2026-09-30T07:16Z case 18 e19_h2_20231025 landed (watchdog): compare + prune done, df 600G -> 600G
- job 2066 on nucla3m-a3meganodeset-3, wall 279.0 min; log_Z -0.826, founders 9, normalization_check 0.911, ESS by step [32.0,16.7,15.9,11.2,18.8]
- 2026-09-30T07:16Z watchdog: running 2020_24@n3 2020_23@n0 2020_21@n6 2020_20@n2 2020_19@n1 ; pending 2020_25 2020_26 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 600G

## 2026-09-30T07:33Z case 23 e24_c3_20240119 landed (watchdog): compare + prune done, df 645G -> 645G
- job 2074 on nucla3m-a3meganodeset-0, wall 31.6 min; log_Z -1.203, founders 8, normalization_check 0.519, ESS by step [32.0,15.4,14.9,15.2,14.2]

## 2026-09-30T07:45Z case 24 e25_h3_20240131 landed (watchdog): compare + prune done, df 658G -> 658G
- job 2075 on nucla3m-a3meganodeset-3, wall 30.5 min; log_Z -0.834, founders 7, normalization_check 0.582, ESS by step [32.0,13.6,22.4,18.6,9.5]
- 2026-09-30T08:21Z watchdog: running 2020_26@n3 2020_25@n0 2020_21@n6 2020_20@n2 2020_19@n1 ; pending 2020_27 2020_28 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 678G

## 2026-09-30T08:59Z case 19 e20_c2_20231103 landed (watchdog): compare + prune done, df 624G -> 624G
- job 2067 on nucla3m-a3meganodeset-1, wall 249.7 min; log_Z -1.937, founders 10, normalization_check 1.255, ESS by step [32.0,13.3,15.8,10.1,20.7]

## 2026-09-30T09:21Z case 20 e21_h2_20231120 landed (watchdog): compare + prune done, df 640G -> 640G
- job 2068 on nucla3m-a3meganodeset-2, wall 246.8 min; log_Z -1.009, founders 4, normalization_check 0.217, ESS by step [32.0,14.6,20.5,17.2,17.4]
- 2026-09-30T09:27Z watchdog: running 2020_28@n2 2020_27@n1 2020_26@n3 2020_25@n0 2020_21@n6 ; pending 2020_29 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 663G

## 2026-09-30T10:20Z case 21 e22_h2_20231210 landed (watchdog): compare + prune done, df 609G -> 608G
- job 2073 on nucla3m-a3meganodeset-6, wall 254.1 min; log_Z -1.755, founders 11, normalization_check 0.930, ESS by step [32.0,16.8,14.7,15.2,24.3]
- 2026-09-30T10:30Z watchdog: running 2020_29@n6 2020_28@n2 2020_27@n1 2020_26@n3 2020_25@n0 ; pending 2020_30 2020_31 2020_32 2020_33 2020_34 2020_35 2020_36 2020_37 2020_38 2020_39 2020_40 2020_41 ; free [none]; foreign [4 5 7]; df 631G
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_30 (case 30 e31_h2_20241027)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_31 (case 31 e32_h2_20241221)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_32 (case 32 e33_h4_20250101)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_33 (case 33 e34_c4_20250125)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_34 (case 34 e35_h2_20250209)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_35 (case 35 e36_c3_20250223)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_36 (case 36 e37_h3_20250302)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_37 (case 37 e38_h2_20250317)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_38 (case 38 e39_h2_20250331)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_39 (case 39 e40_h2_20250930)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_40 (case 40 e41_h2_20251120)
- 2026-09-30T10:38Z acal_ctl (ubuntu): held pending 2020_41 (case 41 e42_h4_20251228)
- 2026-09-30T10:38Z acal_ctl (ubuntu): PAUSED (drain): 17 task(s) handled; resume with: bash slurm/acal_ctl.sh resume

## 2026-09-30T11:59Z case 25 e26_h2_20240211 landed (watchdog): compare + prune done, df 588G -> 588G
- job 2076 on nucla3m-a3meganodeset-0, wall 268.1 min; log_Z -0.238, founders 7, normalization_check 0.709, ESS by step [32.0,15.1,11.2,12.9,7.5]

## 2026-09-30T12:21Z case 26 e27_h2_20240228 landed (watchdog): compare + prune done, df 612G -> 612G
- job 2077 on nucla3m-a3meganodeset-3, wall 272.9 min; log_Z -1.295, founders 5, normalization_check 0.157, ESS by step [32.0,14.2,9.3,15.1,14.7]

## 2026-09-30T12:58Z case 27 e28_h2_20240318 landed (watchdog): compare + prune done, df 641G -> 641G
- job 2078 on nucla3m-a3meganodeset-1, wall 243.4 min; log_Z -0.214, founders 9, normalization_check 2.910, ESS by step [32.0,9.2,18.1,16.9,5.3]

## 2026-09-30T13:36Z case 28 e29_h2_20240418 landed (watchdog): compare + prune done, df 650G -> 650G
- job 2079 on nucla3m-a3meganodeset-2, wall 256.5 min; log_Z -0.424, founders 10, normalization_check 0.622, ESS by step [32.0,18.7,11.7,13.4,13.1]

## 2026-09-30T14:24Z case 29 e30_h2_20241001 landed (watchdog): compare + prune done, df 679G -> 679G
- job 2080 on nucla3m-a3meganodeset-6, wall 240.8 min; log_Z -0.784, founders 5, normalization_check 0.186, ESS by step [32.0,21.2,10.9,21.3,7.7]

## 2026-09-30T23:40Z case 32 e33_h4_20250101 landed (watchdog): compare + prune done, df 605G -> 605G
- job 25512 on phivea3m-a3meganodeset-3, wall 209.9 min; log_Z -1.033, founders 8, normalization_check 0.302, ESS by step [32.0,19.0,17.5,19.0,16.7]

## 2026-09-30T23:41Z CAMPAIGN DONE (watchdog): all 42 cases have res_result.json + compare.json and are pruned

| k | case | summary |
|---|------|---------|
| 0 | e01_h2_20210119 | job 2021 on nucla3m-a3meganodeset-3, wall 308.9 min; log_Z -0.384, founders 9, normalization_check 1.480, ESS by step [32.0,20.2,12.3,9.2,12.4] |
| 1 | e02_c4_20210218 | job 1218 on nucla3m-a3meganodeset-6, wall 245.3 min; log_Z -1.182, founders 5, normalization_check 0.496, ESS by step [32.0,15.9,8.3,16.6,22.2] |
| 2 | e03_h2_20210411 | job 2022 on nucla3m-a3meganodeset-6, wall 286.5 min; log_Z -0.661, founders 8, normalization_check 0.358, ESS by step [32.0,16.4,14.0,10.7,12.3] |
| 3 | e04_h2_20210610 | job 2023 on nucla3m-a3meganodeset-1, wall 245.2 min; log_Z -0.682, founders 7, normalization_check 0.777, ESS by step [32.0,11.0,2.5,10.4,17.0] |
| 4 | e05_h2_20211011 | job 2024 on nucla3m-a3meganodeset-2, wall 259.4 min; log_Z -1.120, founders 8, normalization_check 2.293, ESS by step [32.0,16.8,15.6,9.0,21.1] |
| 5 | e06_h2_20211206 | job 2025 on nucla3m-a3meganodeset-0, wall 310.6 min; log_Z -1.436, founders 5, normalization_check 0.305, ESS by step [32.0,12.2,15.7,10.8,19.9] |
| 6 | e07_h2_20211217 | job 2027 on nucla3m-a3meganodeset-1, wall 240.7 min; log_Z -0.874, founders 7, normalization_check 0.179, ESS by step [32.0,17.4,19.4,7.4,10.0] |
| 7 | e08_h2_20211229 | job 2028 on nucla3m-a3meganodeset-2, wall 249.6 min; log_Z -0.602, founders 9, normalization_check 0.533, ESS by step [32.0,18.1,20.8,17.1,15.8] |
| 8 | e09_c2_20220228 | job 2029 on nucla3m-a3meganodeset-6, wall 260.4 min; log_Z -0.734, founders 8, normalization_check 1.613, ESS by step [32.0,12.5,10.2,16.9,15.3] |
| 9 | e10_c2_20220314 | job 2057 on nucla3m-a3meganodeset-3, wall 286.3 min; log_Z -1.769, founders 4, normalization_check 0.603, ESS by step [32.0,19.6,13.4,4.6,14.9] |
| 10 | e11_h2_20220321 | job 2058 on nucla3m-a3meganodeset-0, wall 276.1 min; log_Z -0.847, founders 8, normalization_check 1.347, ESS by step [32.0,12.4,13.6,17.7,10.9] |
| 11 | e12_c3_20221121 | job 1219 on nucla3m-a3meganodeset-0, wall 223.4 min; log_Z -0.854, founders 11, normalization_check 1.269, ESS by step [32.0,15.1,14.5,14.4,22.0] |
| 12 | e13_c3_20221225 | job 1220 on nucla3m-a3meganodeset-0, wall 238.8 min; log_Z -0.758, founders 7, normalization_check 1.018, ESS by step [32.0,14.9,16.2,11.5,6.8] |
| 13 | e14_h4_20230104 | job 1222 on nucla3m-a3meganodeset-3, wall 142.2 min; log_Z -0.885, founders 8, normalization_check 0.318, ESS by step [32.0,17.1,20.0,20.5,19.4] |
| 14 | e15_h2_20230118 | job 2059 on nucla3m-a3meganodeset-1, wall 239.2 min; log_Z -0.802, founders 9, normalization_check 1.099, ESS by step [32.0,15.8,14.5,16.2,16.8] |
| 15 | e16_c2_20230205 | job 2060 on nucla3m-a3meganodeset-2, wall 243.9 min; log_Z -1.367, founders 9, normalization_check 0.570, ESS by step [32.0,12.0,13.0,19.3,19.4] |
| 16 | e17_c2_20230322 | job 2064 on nucla3m-a3meganodeset-6, wall 258.7 min; log_Z -1.197, founders 9, normalization_check 1.753, ESS by step [32.0,16.7,12.9,15.4,24.3] |
| 17 | e18_h2_20231006 | job 2065 on nucla3m-a3meganodeset-0, wall 270.4 min; log_Z -0.939, founders 8, normalization_check 0.358, ESS by step [32.0,6.3,9.2,16.0,8.5] |
| 18 | e19_h2_20231025 | job 2066 on nucla3m-a3meganodeset-3, wall 279.0 min; log_Z -0.826, founders 9, normalization_check 0.911, ESS by step [32.0,16.7,15.9,11.2,18.8] |
| 19 | e20_c2_20231103 | job 2067 on nucla3m-a3meganodeset-1, wall 249.7 min; log_Z -1.937, founders 10, normalization_check 1.255, ESS by step [32.0,13.3,15.8,10.1,20.7] |
| 20 | e21_h2_20231120 | job 2068 on nucla3m-a3meganodeset-2, wall 246.8 min; log_Z -1.009, founders 4, normalization_check 0.217, ESS by step [32.0,14.6,20.5,17.2,17.4] |
| 21 | e22_h2_20231210 | job 2073 on nucla3m-a3meganodeset-6, wall 254.1 min; log_Z -1.755, founders 11, normalization_check 0.930, ESS by step [32.0,16.8,14.7,15.2,24.3] |
| 22 | e23_h3_20231228 | job 1223 on nucla3m-a3meganodeset-0, wall 218.6 min; log_Z -1.545, founders 10, normalization_check 3.161, ESS by step [32.0,17.2,11.0,14.5,27.1] |
| 23 | e24_c3_20240119 | job 2074 on nucla3m-a3meganodeset-0, wall 31.6 min; log_Z -1.203, founders 8, normalization_check 0.519, ESS by step [32.0,15.4,14.9,15.2,14.2] |
| 24 | e25_h3_20240131 | job 2075 on nucla3m-a3meganodeset-3, wall 30.5 min; log_Z -0.834, founders 7, normalization_check 0.582, ESS by step [32.0,13.6,22.4,18.6,9.5] |
| 25 | e26_h2_20240211 | job 2076 on nucla3m-a3meganodeset-0, wall 268.1 min; log_Z -0.238, founders 7, normalization_check 0.709, ESS by step [32.0,15.1,11.2,12.9,7.5] |
| 26 | e27_h2_20240228 | job 2077 on nucla3m-a3meganodeset-3, wall 272.9 min; log_Z -1.295, founders 5, normalization_check 0.157, ESS by step [32.0,14.2,9.3,15.1,14.7] |
| 27 | e28_h2_20240318 | job 2078 on nucla3m-a3meganodeset-1, wall 243.4 min; log_Z -0.214, founders 9, normalization_check 2.910, ESS by step [32.0,9.2,18.1,16.9,5.3] |
| 28 | e29_h2_20240418 | job 2079 on nucla3m-a3meganodeset-2, wall 256.5 min; log_Z -0.424, founders 10, normalization_check 0.622, ESS by step [32.0,18.7,11.7,13.4,13.1] |
| 29 | e30_h2_20241001 | job 2080 on nucla3m-a3meganodeset-6, wall 240.8 min; log_Z -0.784, founders 5, normalization_check 0.186, ESS by step [32.0,21.2,10.9,21.3,7.7] |
| 30 | e31_h2_20241027 | job 25510 on phivea3m-a3meganodeset-1, wall 262.7 min; log_Z -0.572, founders 9, normalization_check 0.971, ESS by step [32.0,17.1,10.1,19.9,23.9] |
| 31 | e32_h2_20241221 | job 25511 on phivea3m-a3meganodeset-2, wall 248.7 min; log_Z -0.861, founders 7, normalization_check 0.324, ESS by step [32.0,14.6,7.8,20.9,13.5] |
| 32 | e33_h4_20250101 | job 25512 on phivea3m-a3meganodeset-3, wall 209.9 min; log_Z -1.033, founders 8, normalization_check 0.302, ESS by step [32.0,19.0,17.5,19.0,16.7] |
| 33 | e34_c4_20250125 | job 25513 on phivea3m-a3meganodeset-4, wall 255.1 min; log_Z -1.788, founders 10, normalization_check 1.483, ESS by step [32.0,19.7,6.5,7.0,20.9] |
| 34 | e35_h2_20250209 | job 25514 on phivea3m-a3meganodeset-5, wall 246.6 min; log_Z -1.443, founders 8, normalization_check 0.845, ESS by step [32.0,21.7,6.5,18.6,24.1] |
| 35 | e36_c3_20250223 | job 25515 on phivea3m-a3meganodeset-6, wall 251.6 min; log_Z -1.034, founders 7, normalization_check 0.366, ESS by step [32.0,15.2,20.2,18.2,15.7] |
| 36 | e37_h3_20250302 | job 25516 on phivea3m-a3meganodeset-7, wall 249.7 min; log_Z -1.577, founders 9, normalization_check 0.464, ESS by step [32.0,16.7,16.2,14.6,20.2] |
| 37 | e38_h2_20250317 | job 25517 on phivea3m-a3meganodeset-8, wall 251.8 min; log_Z -1.106, founders 7, normalization_check 0.950, ESS by step [32.0,14.4,13.9,12.9,15.2] |
| 38 | e39_h2_20250331 | job 25518 on phivea3m-a3meganodeset-9, wall 246.9 min; log_Z -1.138, founders 9, normalization_check 0.493, ESS by step [32.0,12.9,12.4,21.3,24.4] |
| 39 | e40_h2_20250930 | job 25519 on phivea3m-a3meganodeset-10, wall 246.3 min; log_Z -0.003, founders 6, normalization_check 0.453, ESS by step [32.0,19.1,8.2,9.4,12.6] |
| 40 | e41_h2_20251120 | job 25520 on phivea3m-a3meganodeset-11, wall 254.8 min; log_Z -1.504, founders 6, normalization_check 0.468, ESS by step [32.0,20.0,13.3,16.5,22.1] |
| 41 | e42_h4_20251228 | job 25521 on phivea3m-a3meganodeset-12, wall 250.9 min; log_Z -1.401, founders 5, normalization_check 0.167, ESS by step [32.0,16.8,18.7,16.7,18.4] |

- sacct: 1217_0|COMPLETED|04:05:31|nucla3m-a3meganodeset-6 1217_1|COMPLETED|03:43:43|nucla3m-a3meganodeset-0 1217_2|COMPLETED|03:59:04|nucla3m-a3meganodeset-0 1217_3|FAILED|02:07:03|nucla3m-a3meganodeset-6 1217_4|COMPLETED|03:38:48|nucla3m-a3meganodeset-0 1217_5|CANCELLED by 1000|04:20:42|nucla3m-a3meganodeset-4 1217_6|CANCELLED by 1000|04:05:53|nucla3m-a3meganodeset-2 1217_7|CANCELLED by 1000|00:46:25|nucla3m-a3meganodeset-0 1222_3|COMPLETED|02:22:27|nucla3m-a3meganodeset-3 1250_[0,2-10,14-21,25-31,34,37-40%1]|CANCELLED by 1000|00:00:00|None assigned 1673|CANCELLED by 1000|00:06:09|nucla3m-a3meganodeset-7 2020_0|COMPLETED|05:09:05|nucla3m-a3meganodeset-3 2020_2|COMPLETED|04:46:42|nucla3m-a3meganodeset-6 2020_3|COMPLETED|04:05:24|nucla3m-a3meganodeset-1 2020_4|COMPLETED|04:19:35|nucla3m-a3meganodeset-2 2020_5|COMPLETED|05:10:48|nucla3m-a3meganodeset-0 2020_6|COMPLETED|04:00:52|nucla3m-a3meganodeset-1 2020_7|COMPLETED|04:09:47|nucla3m-a3meganodeset-2 2020_8|COMPLETED|04:20:38|nucla3m-a3meganodeset-6 2020_9|COMPLETED|04:46:30|nucla3m-a3meganodeset-3 2020_10|COMPLETED|04:36:17|nucla3m-a3meganodeset-0 2020_14|COMPLETED|03:59:21|nucla3m-a3meganodeset-1 2020_15|COMPLETED|04:04:08|nucla3m-a3meganodeset-2 2020_16|COMPLETED|04:18:52|nucla3m-a3meganodeset-6 2020_17|COMPLETED|04:30:33|nucla3m-a3meganodeset-0 2020_18|COMPLETED|04:39:10|nucla3m-a3meganodeset-3 2020_19|COMPLETED|04:09:52|nucla3m-a3meganodeset-1 2020_20|COMPLETED|04:06:59|nucla3m-a3meganodeset-2 2020_21|COMPLETED|04:14:17|nucla3m-a3meganodeset-6 2020_23|COMPLETED|00:32:02|nucla3m-a3meganodeset-0 2020_24|COMPLETED|00:30:58|nucla3m-a3meganodeset-3 2020_25|COMPLETED|04:28:17|nucla3m-a3meganodeset-0 2020_26|COMPLETED|04:33:04|nucla3m-a3meganodeset-3 2020_27|COMPLETED|04:03:38|nucla3m-a3meganodeset-1 2020_28|COMPLETED|04:16:44|nucla3m-a3meganodeset-2 2020_29|COMPLETED|04:01:02|nucla3m-a3meganodeset-6 
- df /home: 634G free
