# TileLang File Review Scenario

## Triggers

Review code, audit code, check coding standards, perform a code review, or review specified files for me.

---

## Orchestration

### Task List

At startup, create three fixed tasks, all marked `pending`:

| Task | Stage | Content |
|---|---|---|
| Task 0 | Code Summary + Review Plan Design | Run code-summarize first, then plan-design |
| Task 1 | Format Check + Clause-by-Clause Review | Run the format check and first batch of clauses in parallel; after each task completes and its result is confirmed, dispatch the next queued task |
| Task 2 | Report Writing | Execute `steps/common.report-write.md` |

### Input Parsing

Extract the files to review from user input and normalize them as `file_input`. Supported inputs include a single file path, multiple file paths, and a file list obtained by enumerating a directory. Python, TileLang DSL, and Markdown are all valid inputs.

Process only review targets that can be determined from user input. Do not expand the file set independently.

### Stage 0: Code Summary + Review Plan Design

1. Mark Task 0 as `in_progress`.
2. Extract `operator_name` from `file_input` and use it as the review identifier for this workflow. Following the "Artifact and Resource Paths" rules in `SKILL.md`, establish the absolute path `review_output_dir` before switching shell directories or dispatching a subagent. If the user specifies a relative output directory, resolve it against the user's working directory at invocation time. Also determine the absolute path `skill_base` from the actual location of this `SKILL.md`; use it only to access Skill resources.
3. Use `spawn_agent` to dispatch a code-summary subagent that reads and executes `steps/file-review.code-summarize.md`. Pass `file_input` and the summary output path `{review_output_dir}/code_summary.md`. It returns the file type, code side, summary path, and cross-file relationships. Determine the side of Python and TileLang code from the actual input as Kernel, Host, or mixed; the side of Markdown is `N/A`.
4. Wait for the code-summary subagent to complete, then collect the file type, code side, code-summary path, and cross-file relationships.
5. Read and execute `steps/common.plan-design.md`. Dispatch a separate plan-design subagent to produce general groupings, waves, the expected clause set, and the skip list. Pass:
   - `file_input`
   - File type
   - Code side
   - Code-summary path
   - `scope_hint`
   - Review type `file`
   - An empty review identifier because a file review has no PR number
   - The number of subagent slots available to Stage 1
   - `skill_base` (the plan-design subagent must invoke Skill scripts and read references)

   `plan-design` is responsible for invoking `scripts/workflow.create_review_dir.py` to create the YAML output directory and returning `yaml_dir`.
6. Mark Task 0 as `done`.

### Stage 1: Format Check + Clause-by-Clause Review

1. Mark Task 1 as `in_progress`.
2. Start the YAML collector service. Subagents submit YAML only through HTTP and do not access `yaml_dir`:
   - The collector listens only on the local loopback address. If the environment sets `all_proxy`, `ALL_PROXY`, `http_proxy`, or `https_proxy`, do not clear or overwrite those external proxies. Add local addresses to the proxy bypass list before starting the collector:

     ```bash
     export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
     export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
     ```

     Health checks and all subagent YAML submissions to the collector must still explicitly use `--noproxy 127.0.0.1,localhost` with `curl`, preventing settings such as `all_proxy=socks5://172.17.0.1:33334` from hijacking local HTTP requests. This bypass applies only to the collector and does not prevent external resource downloads by Ruff or other tools from continuing to use existing proxies.
   - In the same shell invocation, select a port, start the collector, immediately record `$!`, and perform at most two readiness checks. Do not append the trailing `&` to a chain of `&&` commands, because that may background the entire chain. After the startup invocation completes, the main agent must retain the printed port and exact PID; later shell invocations must not assume that these variables still exist:

     ```bash
     REVIEW_COLLECTOR_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
     setsid python3 "{skill_base}/scripts/workflow.submit_server.py" "{yaml_dir}" "$REVIEW_COLLECTOR_PORT" > "/tmp/collector_${REVIEW_COLLECTOR_PORT}.log" 2>&1 < /dev/null &
     REVIEW_COLLECTOR_PID=$!
     printf 'collector_port=%s collector_pid=%s\n' "$REVIEW_COLLECTOR_PORT" "$REVIEW_COLLECTOR_PID"
     sleep 1
     if ! curl -sS --noproxy 127.0.0.1,localhost --max-time 3 "http://127.0.0.1:${REVIEW_COLLECTOR_PORT}/health"; then
       sleep 2
       if ! curl -sS --noproxy 127.0.0.1,localhost --max-time 3 "http://127.0.0.1:${REVIEW_COLLECTOR_PORT}/health"; then
         sed -n '1,160p' "/tmp/collector_${REVIEW_COLLECTOR_PORT}.log"
         kill "$REVIEW_COLLECTOR_PID" 2>/dev/null || true
         exit 1
       fi
     fi
     ```

   - If both checks still fail, the newly started process has already been terminated by its exact PID. Diagnose recoverable causes using that run's log, then rerun the startup block and record the new port and PID. Do not dispatch a subagent to connect to a collector that has not passed its health check.
3. Read `steps/common.format-check.md` and `steps/file-review.clause-review.md` to obtain the execution templates for format checking and clause-by-clause review.
4. Preserve the grouping contents and stable priority from the review plan. Flatten the planned waves in their original order into a dispatch queue; planned waves represent only the initial capacity allocation and are not waiting barriers in file review. Schedule continuously in completion order:
   - At the start of Stage 1, set the concurrency limit to `min(review slots available for this run, 6)`. The total number of running format-check and clause-review subagents must not exceed this limit. Before every dispatch, also confirm that the environment currently has a free subagent slot; do not over-dispatch based only on the Stage 0 slot count.
   - In the first batch, dispatch one format-check subagent in parallel with at most `min(number of clause groups not yet dispatched, review slots available for this run - 1, 5)` clause-review subagents. Confirm actual free slots before every dispatch. If only one slot is available, run the format check by itself first. The format check remains mandatory when there are no clause groups.
   - The format-check subagent reads and executes `steps/common.format-check.md`, receives `file_input`, `collector_port`, and `skill_base`, and submits one aggregate YAML document through `/submit?type=format`.
   - Clause groups that do not fit in the first batch remain in the queue. After a task completes and passes its corresponding result check from Step 6, immediately dispatch the next group in queue order if one remains and a slot is available; do not wait for other concurrently running tasks. Do not interrupt a lower-priority task that is already running when a later task finishes.
   - Invoke `spawn_agent` once for each format or clause task. Keep every `task_name` unique within this review. Use a new subagent for every dispatch; do not reuse a completed subagent for another group.
   - Following the message template, every clause-review subagent receives the group identifier, file type, code side, reference filename, original clause ID and title, `file_input`, code-summary path, and `collector_port`.
   - Do not pass `yaml_dir` to any subagent, and do not allow subagents to read results already stored there.
   - Each clause-review subagent reviews every clause independently and submits YAML through the following endpoint, where `rule` is the reference filename with or without the `.md` suffix:

     ```text
     http://127.0.0.1:{recorded_collector_port}/submit?group={group_id}&rule={reference_file}&clause={clause_id}
     ```

     The collector forcibly writes `group_name`, `rule_file`, `canonical_id`, and `submission_key`. When the same identity is submitted more than once, it preserves the existing result and creates a new file with a `_dup1`, `_dup2`, and subsequent suffix.

5. After each task completes and its result is checked, output completed/total counts and the current running count. Once the queue is fully dispatched and all running tasks have returned, summarize completion. Treat YAML written by the collector as authoritative; do not collect details again from subagent text responses.
6. After each subagent returns, verify that task's expected results before releasing the slot and dispatching the next group:
   - Build the expected YAML set from the clause group and its clause IDs. Do not use "the YAML count equals the agent count" as the completeness criterion; check that a result exists for every expected clause in the group.
   - When the format task completes, confirm that at least one parseable aggregate YAML document exists with top-level `type: format`. `ERROR` and `PARTIAL` are valid format results and do not trigger redispatch based on status.
   - If a clause group or format result is confirmed missing, automatically redispatch the corresponding complete task once, prioritizing it for the freed slot. The retry task must also use a new subagent and a unique `task_name`; two instances of the same group must never run concurrently. If the result remains missing after the retry, stop dispatching new groups, wait for running tasks to return, terminate this run's collector, and report the missing items. Do not retry indefinitely.
   - After all tasks return, perform one global completeness check against the full expected clause set from the plan and the format result. Apply the same one-retry limit to any missing item. Terminate the collector and proceed to the report stage only after all results are confirmed.
7. After all tasks complete and pass the global completeness check, terminate this run's collector using the recorded exact PID:

   ```bash
   kill {recorded_exact_COLLECTOR_PID}
   ```

   Use the exact PID printed by the startup invocation and first verify that it belongs to this run's collector. Before exiting, terminate this run's collector whether the stage succeeds, fails, or is interrupted. Do not terminate other instances by process name.
8. After confirming that YAML for every expected clause and the format result is stored and the collector is terminated, mark Task 1 as `done`.

### Stage 2: Report Writing

1. Mark Task 2 as `in_progress`.
2. Read and execute `steps/common.report-write.md`. Pass the Stage 0 `review_output_dir`, `yaml_dir`, report output path, and header metadata such as file type, code side, and timestamp. This step invokes `scripts/workflow.assemble_report.py` to assemble the report body, after which the main agent completes the remaining header information.
3. The report output path is `{review_output_dir}/{source_file}_review_summary.md`; pass this absolute path to the assembler. Do not rederive the artifact location from a later shell working directory, the input repository, or the Skill installation location.
4. After replacing the header information, check the report's line count. If it exceeds 5,000 lines, read and execute `steps/common.report-filter.md`, using the AscendC approach to remove non-severe clause findings by severity, compress format details, and update statistics.
5. Mark Task 2 as `done`.

---

## Context-Passing Chain

```text
Stage 0 → code-summarize → file type + code side + summary path + cross-file relationships
                              ↓
          plan-design → review plan + yaml_dir
                              ↓
Stage 1 → format-check subagent + first-batch clause subagents submit YAML in parallel
                              ↓
          subsequent general-review subagents continuously fill freed slots by priority and submit YAML
                              ↓
Stage 2 → assemble_report.py reads yaml_dir and assembles the final report
```

## Constraints

- Execute stages strictly in order; do not skip steps.
- Stage 0 plan-design must wait for code-summarize to return before being dispatched separately.
- This workflow does not perform API preliminary research and does not pass an API preliminary-research report path.
- This workflow does not count lines of code, invoke `workflow.review_mode.py`, or produce or pass `mode` or `guidance`.
- The format-check subagent counts toward both the maximum of six concurrent subagents and the current available-slot concurrency limit. The format check produces results only and does not apply formatting fixes.
- After the collector starts, its recorded exact PID must be terminated before leaving Stage 1, whether the stage succeeds, fails, or is interrupted.
- Do not read step files for stages that have not yet begun.
