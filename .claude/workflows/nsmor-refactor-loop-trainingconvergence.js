export const meta = {
  name: 'nsmor-refactor-loop',
  description: 'NSMoR training stability refactor with parallel double-blind review',
  phases: [
    { title: 'Diagnose', detail: 'Developer analyzes training instability and proposes fixes' },
    { title: 'Review', detail: 'Parallel blind review by Reviewer A and Reviewer B' },
    { title: 'Test', detail: 'Run training to validate fixes' },
  ],
}

const WSL_PYTHON = `
IMPORTANT: All Python commands must be run via WSL zsh. Use this exact pattern:
\`\`\`bash
wsl zsh -c "source ~/.zshrc && openconda && conda activate torch && python <your_command>"
\`\`\`
Do NOT use bare python. Always go through WSL zsh with openconda + conda activate torch.
`

const TRAINING_LOG = `
[2026-06-29 21:25:17] INFO — Epoch 10/100  train_loss=2.963212  val_loss=3.940143  time=10.4s
[2026-06-29 21:25:27] INFO — Epoch 11/100  train_loss=2.635703  val_loss=2.015475  time=10.4s
[2026-06-29 21:25:38] INFO — Epoch 12/100  train_loss=4.278736  val_loss=3.090133  time=10.5s
[2026-06-29 21:25:48] INFO — Epoch 13/100  train_loss=2.548094  val_loss=2.357962  time=10.4s
[2026-06-29 21:25:59] INFO — Epoch 14/100  train_loss=2.840863  val_loss=1.821872  time=10.7s
[2026-06-29 21:26:10] INFO — Epoch 15/100  train_loss=2.204310  val_loss=3.563010  time=10.6s
[2026-06-29 21:26:20] INFO — Epoch 16/100  train_loss=2.643279  val_loss=2.620360  time=10.4s
[2026-06-29 21:26:30] INFO — Epoch 17/100  train_loss=2.601823  val_loss=1.538217  time=9.9s
[2026-06-29 21:26:41] INFO — Epoch 18/100  train_loss=3.621799  val_loss=3.587430  time=10.9s
[2026-06-29 21:26:51] INFO — Epoch 19/100  train_loss=1.867539  val_loss=5.395702  time=10.2s
[2026-06-29 21:27:02] INFO — Epoch 20/100  train_loss=2.247000  val_loss=1.534042  time=10.5s
[2026-06-29 21:27:12] INFO — Epoch 21/100  train_loss=2.902909  val_loss=2.048024  time=10.0s
[2026-06-29 21:27:23] INFO — Epoch 22/100  train_loss=2.594735  val_loss=3.147583  time=11.2s
[2026-06-29 21:27:33] INFO — Epoch 23/100  train_loss=3.141294  val_loss=4.041363  time=10.4s
[2026-06-29 21:27:45] INFO — Epoch 24/100  train_loss=1.275846  val_loss=2.823176  time=11.3s
[2026-06-29 21:27:56] INFO — Epoch 25/100  train_loss=2.603765  val_loss=3.310281  time=11.4s
[2026-06-29 21:28:07] INFO — Epoch 26/100  train_loss=2.313901  val_loss=2.470676  time=10.9s
[2026-06-29 21:28:18] INFO — Epoch 27/100  train_loss=3.916029  val_loss=5.918972  time=11.6s
[2026-06-29 21:28:30] INFO — Epoch 28/100  train_loss=2.688052  val_loss=4.201823  time=11.1s
[2026-06-29 21:28:40] INFO — Epoch 29/100  train_loss=1.843243  val_loss=2.130566  time=10.7s
`

const DEVELOPER_PROMPT = `
You are the NSMoR developer agent. Your task: diagnose training instability and produce a concrete refactoring plan with code changes.

## Current Training Log (train_loss oscillates 1.28–4.28, val_loss 1.53–5.92, NO convergence)
${TRAINING_LOG}

## Architecture Summary
- Dual-pathway RNN: LIF spiking neurons + GRU, blended by MoR Router
- BioJointLoss = Masked MSE + Router Reg + ATP Energy + Pop Sparsity L1 + Jerk penalty
- Default config: lr=0.001, no LR scheduler, grad_clip=1.0.0, batch=32
- bio-loss lambdas (energy/sparse/jerk) all default to 0.0 (disabled)
- Validation uses FULL lambdas, training uses warmup scaling

## Key Code Files
- Model: nsmor/model_nsmor_core.py (~1960 lines)
- Loss: nsmor/loss.py (~503 lines)
- Training: scripts/train.py (~868 lines)
- Config: config/default.yaml + nsmor/config_parser.py

## Root Cause Analysis (your job to confirm/refine)
1. **No LR scheduler**: lr=0.001 stays constant → optimizer overshoots as loss landscape changes
2. **LIF pathway instability**: Discrete spike events + surrogate gradient estimator produce noisy gradient signals. The spike on/off boundary creates discontinuous loss landscape.
3. **Train/val mismatch**: Training uses warmup_factor (0→1) for bio-losses, validation uses full lambdas → val_loss artificially inflated early, biased best_model selection
4. **No warmup for main MSE loss**: Cold start with full lr on a complex biophysical model
5. **Router gate instability**: Router's sigmoid output can oscillate between LIF and GRU pathways

## Constraints
- **DO NOT modify nsmor/model_nsmor_core.py** (frozen core, per .claude.md)
- **DO NOT modify nsmor/loss.py** (frozen core, per .claude.md)
- Changes MUST be in scripts/train.py and/or config/default.yaml
- Must maintain backward compatibility (all existing tests must pass)
- Python environment: WSL zsh, \`openconda\`, \`conda activate torch\`

## Output Format
Produce a JSON-structured refactoring plan with:

\`\`\`json
{
  "diagnosis": {
    "root_causes": ["cause1", "cause2", ...],
    "severity_ranking": ["most_severe", ...]
  },
  "changes": [
    {
      "file": "scripts/train.py",
      "what": "description of change",
      "why": "biophysical/engineering justification",
      "code_diff": "the actual code to add/modify (be precise)"
    }
  ],
  "config_changes": [
    {
      "file": "config/default.yaml",
      "what": "description",
      "code_diff": "the actual yaml change"
    }
  ],
  "expected_outcome": "what the training curve should look like after fixes"
}
\`\`\`

Focus on changes that are:
1. Minimal but effective (least code change, maximum stability improvement)
2. Biologically principled (not just engineering hacks)
3. Backward compatible (existing tests pass)
`

const REVIEWER_TEMPLATE = (reviewerId) => `
You are NSMoR Reviewer ${reviewerId} — an independent, adversarial code reviewer with expertise in:
- Computational neuroscience (spiking neural networks, biophysical modeling)
- Deep learning optimization (gradient stability, loss landscape analysis)
- PyTorch internals (autograd, gradient flow, numerical stability)

Your job: REVIEW the proposed refactoring plan with SKEPTICAL rigor. You must evaluate BOTH:
1. **Biophysical correctness** — Does the change preserve biological fidelity?
2. **Engineering soundness** — Does the fix address the actual root cause?

## Current Training Log
${TRAINING_LOG}

## Architecture Context
- Dual-pathway: LIF (spiking, discrete) + GRU (continuous, smooth)
- MoR Router blends them via sigmoid gates
- BioJointLoss = MSE + router_reg + energy + sparsity + jerk
- LIF uses surrogate gradient: spike = mask - sigmoid.detach() + sigmoid

## Constraints You Must Enforce
- nsmor/model_nsmor_core.py is FROZEN — any proposal modifying it must be REJECTED
- nsmor/loss.py is FROZEN — any proposal modifying it must be REJECTED
- All existing tests must still pass
- Changes must be backward compatible

## Your Review Checklist
For EACH proposed change, verify:

1. **Root cause alignment**: Does this change actually fix the identified instability cause?
2. **Gradient flow**: Could this change create gradient dead zones or explosion?
3. **Biological fidelity**: Does this violate any biological constraint from the references?
4. **Numerical stability**: Are there edge cases (zero division, NaN, overflow)?
5. **Test compatibility**: Will existing tests in tests/test_biophysics.py, tests/test_loss.py, tests/test_pipeline.py still pass?
6. **Backward compatibility**: Does this break the public API?

## Output Format
\`\`\`json
{
  "reviewer_id": "${reviewerId}",
  "is_accepted": true/false,
  "verdict": "ACCEPT" or "REJECT",
  "overall_assessment": "one paragraph summary",
  "per_change_reviews": [
    {
      "change_description": "what the change does",
      "verdict": "APPROVE" or "REJECT" or "REQUEST_MODIFICATION",
      "reasoning": "detailed reasoning",
      "suggestions": ["improvement1", "improvement2"]
    }
  ],
  "critical_issues": ["issue1 that blocks acceptance", ...],
  "minor_suggestions": ["suggestion1", ...]
}
\`\`\`

Be THOROUGH. If you find even ONE critical issue, you MUST reject.
Do NOT rubber-stamp. Your value is in finding problems the developer missed.
`

const TESTER_PROMPT = (developerOutput) => `
You are the NSMoR tester agent. Execute the proposed code changes and run the training to verify stability.

${WSL_PYTHON}

## Steps
1. Apply ALL code changes from the developer's plan to the actual files
2. Run existing tests to verify backward compatibility:
   \`\`\`bash
   wsl zsh -c "source ~/.zshrc && openconda && conda activate torch && cd /mnt/d/Projects/NSMoR && python -m pytest tests/ -v --tb=short 2>&1 | head -100"
   \`\`\`
3. Run training for 30 epochs to verify stability:
   \`\`\`bash
   wsl zsh -c "source ~/.zshrc && openconda && conda activate torch && cd /mnt/d/Projects/NSMoR && python scripts/train.py --config config/default.yaml --epochs 30 2>&1"
   \`\`\`
4. Analyze the training curve:
   - train_loss should show monotonic decrease or stable oscillation within ±20%
   - val_loss should show downward trend
   - No epochs with train_loss > 5.0 or val_loss > 8.0

## Developer's Changes
${developerOutput}

## Output Format
\`\`\`json
{
  "tests_passed": true/false,
  "test_output": "relevant test output",
  "training_completed": true/false,
  "training_log": "the full training log output",
  "stability_analysis": {
    "train_loss_range": [min, max],
    "val_loss_range": [min, max],
    "is_stable": true/false,
    "convergence_trend": "improving/stagnant/diverging"
  },
  "verdict": "PASS" or "FAIL",
  "failure_reason": "if FAIL, what went wrong"
}
\`\`\`

IMPORTANT: Do NOT modify nsmor/model_nsmor_core.py or nsmor/loss.py.
Only modify scripts/train.py and config/default.yaml.
`

// ── Main Workflow ──────────────────────────────────────────────

const MAX_REVIEW_ROUNDS = 3

phase('Diagnose')
log('NSMoR Training Stability Refactor — Starting double-blind review loop')

const developerOutput = await agent(
  DEVELOPER_PROMPT,
  {
    label: 'nsmor_developer',
    phase: 'Diagnose',
    effort: 'high',
  }
)

log('Developer produced refactoring plan. Launching parallel blind review...')

// ── Review Loop ────────────────────────────────────────────────

let round = 1
let accepted = false
let lastDevOutput = developerOutput

while (round <= MAX_REVIEW_ROUNDS && !accepted) {
  phase('Review')
  log(`Review round ${round}/${MAX_REVIEW_ROUNDS} — launching parallel blind reviewers`)

  // Parallel blind review: A and B are independent, cannot see each other
  const [reviewA, reviewB] = await parallel([
    () => agent(
      REVIEWER_TEMPLATE('A'),
      { label: 'reviewer_A', phase: 'Review', effort: 'high' }
    ),
    () => agent(
      REVIEWER_TEMPLATE('B'),
      { label: 'reviewer_B', phase: 'Review', effort: 'high' }
    ),
  ])

  // Parse verdicts
  let aAccepted = false
  let bAccepted = false
  let aAssessment = ''
  let bAssessment = ''
  let aCritical = []
  let bCritical = []

  try {
    const aResult = typeof reviewA === 'string' ? JSON.parse(reviewA) : reviewA
    aAccepted = aResult.verdict === 'ACCEPT' || aResult.is_accepted === true
    aAssessment = aResult.overall_assessment || ''
    aCritical = aResult.critical_issues || []
  } catch {
    // If parsing fails, try to extract verdict from text
    aAccepted = typeof reviewA === 'string' && reviewA.includes('"verdict": "ACCEPT"')
    aAssessment = typeof reviewA === 'string' ? reviewA.substring(0, 500) : 'Parse error'
  }

  try {
    const bResult = typeof reviewB === 'string' ? JSON.parse(reviewB) : reviewB
    bAccepted = bResult.verdict === 'ACCEPT' || bResult.is_accepted === true
    bAssessment = bResult.overall_assessment || ''
    bCritical = bResult.critical_issues || []
  } catch {
    bAccepted = typeof reviewB === 'string' && reviewB.includes('"verdict": "ACCEPT"')
    bAssessment = typeof reviewB === 'string' ? reviewB.substring(0, 500) : 'Parse error'
  }

  if (aAccepted && bAccepted) {
    accepted = true
    log(`✅ Round ${round}: BOTH reviewers ACCEPTED. Proceeding to testing.`)
  } else {
    // Merge feedback and send back to developer
    const rejectedBy = []
    if (!aAccepted) rejectedBy.push('Reviewer A')
    if (!bAccepted) rejectedBy.push('Reviewer B')

    log(`❌ Round ${round}: REJECTED by ${rejectedBy.join(' and ')}. Sending feedback to developer.`)

    if (!aAccepted) log(`  Reviewer A critical issues: ${aCritical.join('; ') || aAssessment}`)
    if (!bAccepted) log(`  Reviewer B critical issues: ${bCritical.join('; ') || bAssessment}`)

    // Send merged feedback to developer for next round
    const mergedFeedback = `
## Review Round ${round} — REJECTED

### Reviewer A Assessment
Verdict: ${aAccepted ? 'ACCEPT' : 'REJECT'}
${aAssessment}
Critical issues: ${JSON.stringify(aCritical)}

### Reviewer B Assessment
Verdict: ${bAccepted ? 'ACCEPT' : 'REJECT'}
${bAssessment}
Critical issues: ${JSON.stringify(bCritical)}

## Your Task
Address ALL critical issues from both reviewers. Produce an updated refactoring plan.
Apply the same JSON output format as before.
`

    lastDevOutput = await agent(
      mergedFeedback + '\n\n' + DEVELOPER_PROMPT,
      {
        label: `nsmor_developer_r${round + 1}`,
        phase: 'Diagnose',
        effort: 'high',
      }
    )

    round++
  }
}

if (!accepted) {
  log(`⛔ Failed after ${MAX_REVIEW_ROUNDS} review rounds. Last developer output:\n${typeof lastDevOutput === 'string' ? lastDevOutput.substring(0, 2000) : JSON.stringify(lastDevOutput).substring(0, 2000)}`)
  return { status: 'FAILED', reason: 'Review loop exhausted without acceptance', rounds: round }
}

// ── Testing Phase ──────────────────────────────────────────────

phase('Test')
log('Both reviewers accepted. Launching tester to apply changes and validate...')

const testResult = await agent(
  TESTER_PROMPT(typeof lastDevOutput === 'string' ? lastDevOutput : JSON.stringify(lastDevOutput)),
  {
    label: 'nsmor_tester',
    phase: 'Test',
    effort: 'high',
  }
)

let testPassed = false
let testOutput = ''
try {
  const tResult = typeof testResult === 'string' ? JSON.parse(testResult) : testResult
  testPassed = tResult.verdict === 'PASS'
  testOutput = JSON.stringify(tResult, null, 2)
} catch {
  testPassed = typeof testResult === 'string' && testResult.includes('"verdict": "PASS"')
  testOutput = typeof testResult === 'string' ? testResult : 'Parse error'
}

if (!testPassed) {
  log(`❌ Testing FAILED. Sending error log back to developer.`)

  // One retry with test feedback
  const devFix = await agent(
    `The tester found issues. Fix them and produce updated changes.\n\nTest output:\n${testOutput}\n\n${DEVELOPER_PROMPT}`,
    { label: 'nsmor_developer_fix', phase: 'Diagnose', effort: 'high' }
  )

  // Re-test
  const retestResult = await agent(
    TESTER_PROMPT(typeof devFix === 'string' ? devFix : JSON.stringify(devFix)),
    { label: 'nsmor_tester_retry', phase: 'Test', effort: 'high' }
  )

  let retestPassed = false
  try {
    const r = typeof retestResult === 'string' ? JSON.parse(retestResult) : retestResult
    retestPassed = r.verdict === 'PASS'
  } catch {
    retestPassed = typeof retestResult === 'string' && retestResult.includes('"verdict": "PASS"')
  }

  if (!retestPassed) {
    log(`⛔ Retest also failed. Manual intervention required.`)
    return { status: 'FAILED', reason: 'Testing failed after retry', testOutput: typeof retestResult === 'string' ? retestResult : JSON.stringify(retestResult) }
  }
}

// ── Git Commit & Push ──────────────────────────────────────────

log('✅ All tests passed. Committing changes...')

// Note: git operations done by the main agent, not sub-agents
return {
  status: 'SUCCESS',
  rounds: round,
  message: 'Training stability refactor completed. All reviews passed, tests passed.',
  developerPlan: typeof lastDevOutput === 'string' ? lastDevOutput : JSON.stringify(lastDevOutput),
  testResult: testOutput,
}
