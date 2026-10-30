#!/usr/bin/env Rscript
# are-eval.R: Posit's `are` R-coding eval (vitals, 29 problems) against an OpenAI-compatible server.
#
# Two steps, because they need different things:
#   solve  talks only to the local server and needs no key. It is the slow part (hours on a
#          thinking model), so it saves every answer to <out>/<label>-answers.rds as it goes and
#          resumes from that file.
#   score  grades the saved answers with vitals' model_graded_qa(partial_credit = TRUE), the
#          scorer Posit used for its published R results, so it needs a grader model. By default
#          that is Claude (ANTHROPIC_API_KEY in the environment); pass --grader-url/--grader-model
#          to grade with another OpenAI-compatible server instead. Never grade a model with itself.
#
#   Rscript are-eval.R solve --base http://127.0.0.1:8088/v1 --model ornith-1.5-35b-a3b \
#       --label ornith35 --epochs 3 --out ~/models/groot-19b/gate/are
#   ANTHROPIC_API_KEY=... Rscript are-eval.R score --label ornith35 --out ~/models/groot-19b/gate/are
#
# Answers are the final text only: llama-server returns the reasoning separately and ellmer keeps it
# out of the reply, which is what a user in RStudio sees. The server's own sampler applies (no
# temperature is sent), so the unit's shipped settings are what gets measured.
suppressPackageStartupMessages({
  library(ellmer)
  library(vitals)
})

args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
opt <- function(name, default = NULL) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) default else args[i + 1]
}
label <- opt("label")
out <- path.expand(opt("out", "are-results"))
if (is.na(mode) || is.null(label)) stop("usage: are-eval.R solve|score --label NAME [--out DIR] ...")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
answers_file <- file.path(out, paste0(label, "-answers.rds"))

# A thinking model can take minutes on one problem; ellmer's default timeout is far shorter.
options(ellmer_timeout_s = 3600)

if (mode == "solve") {
  base <- opt("base")
  model <- opt("model")
  epochs <- as.integer(opt("epochs", "3"))
  max_tokens <- as.integer(opt("max-tokens", "32768"))
  done <- if (file.exists(answers_file)) readRDS(answers_file) else NULL
  for (e in seq_len(epochs)) {
    for (k in seq_len(nrow(are))) {
      id <- are$id[k]
      if (!is.null(done) && any(done$id == id & done$epoch == e)) next
      chat <- chat_openai_compatible(
        base, model = model, credentials = function() "none",
        params = params(max_tokens = max_tokens)
      )
      t0 <- Sys.time()
      res <- tryCatch(chat$chat(are$input[k], echo = "none"), error = function(err) {
        message("   error on ", id, ": ", conditionMessage(err))
        NA_character_
      })
      secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      tok <- tryCatch(chat$get_tokens(), error = function(err) NULL)
      row <- data.frame(
        id = id, epoch = e, answer = as.character(res), seconds = round(secs, 1),
        output_tokens = if (is.null(tok)) NA_real_ else sum(tok$output),
        stringsAsFactors = FALSE
      )
      done <- rbind(done, row)
      saveRDS(done, answers_file)
      cat(sprintf(">> epoch %d  %-36s %6.1f s  %6s tok\n", e, id, secs, row$output_tokens))
    }
  }
  cat(sprintf("solved %d answers -> %s\n", nrow(done), answers_file))
} else if (mode == "score") {
  done <- readRDS(answers_file)
  grader_url <- opt("grader-url")
  grader <- if (is.null(grader_url)) {
    chat_anthropic(model = opt("grader-model", "claude-sonnet-5-5"), echo = "none")
  } else {
    # Thinking off by default: a local thinking grader spent ~8,500 tokens (6.5 min at 22 t/s) per
    # verdict, and vitals sends grades in parallel, so a single-slot server timed most of them out
    # (2026-10-29, Qwen3.8-27B). --grader-think keeps it on.
    chat_openai_compatible(
      grader_url, model = opt("grader-model"), credentials = function() "none",
      params = params(max_tokens = 4096),
      api_args = if ("--grader-think" %in% args) list() else
        list(chat_template_kwargs = list(enable_thinking = FALSE))
    )
  }
  epochs <- max(done$epoch)
  # Rebuild each saved exchange as a chat, so vitals' scorer and its log see exactly what was said.
  # The solver replays them in the order Task$eval() asks for: the dataset repeated per epoch.
  replay <- function(inputs, ...) {
    # Map each input back to its problem by text, and count repeats for the epoch, rather than
    # trusting the order vitals hands them over in.
    ids <- are$id[match(unlist(inputs), are$input)]
    ep <- ave(seq_along(ids), ids, FUN = seq_along)
    chats <- vector("list", length(inputs))
    results <- character(length(inputs))
    for (i in seq_along(inputs)) {
      a <- done$answer[done$id == ids[i] & done$epoch == ep[i]]
      a <- if (length(a) == 0 || is.na(a[1])) "" else a[1]
      ch <- chat_openai_compatible("http://127.0.0.1:1/v1", model = label, credentials = function() "none")
      ch$set_turns(list(UserTurn(list(ContentText(inputs[[i]]))), AssistantTurn(list(ContentText(a)))))
      chats[[i]] <- ch
      results[i] <- a
    }
    list(result = results, solver_chat = chats)
  }
  vitals_log_dir_set(file.path(out, "logs"))
  task <- Task$new(
    dataset = are, solver = replay,
    scorer = model_graded_qa(partial_credit = TRUE, scorer_chat = grader),
    name = paste0("are-", label)
  )
  task$eval(epochs = epochs, view = FALSE)
  s <- task$get_samples()
  # An empty submission is a fail whatever the grader says: Qwen3.8-27B graded one C (2026-10-29).
  # Empty answers come from runs that hit --max-tokens while still thinking.
  empty <- is.na(s$result) | !nzchar(trimws(s$result))
  if (any(empty)) s$score[empty] <- "I"
  pts <- c(C = 1, P = 0.5, I = 0)[as.character(s$score)]
  summary <- data.frame(
    label = label, n = nrow(s), C = sum(s$score == "C"), P = sum(s$score == "P"),
    I = sum(s$score == "I"), points = sum(pts), pct = round(100 * mean(pts), 1),
    median_seconds = median(done$seconds), median_output_tokens = median(done$output_tokens, na.rm = TRUE)
  )
  write.csv(as.data.frame(s[, c("id", "epoch", "score")]), file.path(out, paste0(label, "-scores.csv")), row.names = FALSE)
  print(summary)
  cat(jsonlite::toJSON(as.list(summary), auto_unbox = TRUE), "\n",
      file = file.path(out, "summary.jsonl"), append = TRUE)
} else {
  stop("mode must be solve or score")
}
