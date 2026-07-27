You are working on repo wray-lee/NSMoR. FULL AUDIT REQUIRED BEFORE CODING. Read Makefile, config/default.yaml, nsmor/config.py, config_parser.py, nsmor/model_nsmor_core.py, nsmor/analysis/dynamics.py, model_utils.py, checkpoint.py, nsmor_dataloader.py, nsmor/pipeline/labeling.py, data_extractor.py, requirements.txt/pyproject.toml, Dockerfile,.github/workflows.

CONTEXT: Makefile: load, data, train, analyze (currently 5), dynamics, lesion, jacobian, integration, psychophysics, generate, test, pipeline. config/default.yaml is single source of truth backed by frozen dataclasses. No hardcoded numbers. NSMoRCore forward(return_internals=True) -> routing_gates [B,T,2]. Sequences are Trial-Start anchored. TTC-50ms is ONLY for MCMC 5-D snapshot, NOT for gating alignment. labeling.py is 4-way: Pre_Active, Startle, Walk, NoResponse. TimeWindowConfig(baseline 5700ms) is VARIANT for pure-wind only. CI: checkout -> make install py3.10 -> make test. Docker: docker compose run --rm nsmor analyze.

GOAL: Add 6th analysis: unsupervised gating strategy clustering, window-free by design.

IMPLEMENTATION:

1. nsmor/analysis/gating_cluster.py
    - class GatingClusterAdapter: extract_gating_sequences() eval mode, torch.no_grad(), shuffle=False, un-pad via lengths, return {trial_id, gates[T,2], true_4way, true_3way_merged}. Mapping for EVAL ONLY: Startle->Escape, Walk->PreWalk, Pre_Active->PreWalk, NoResponse->NoResponse.
    - def fingerprint(gates[T,2], config: ClusterGatingConfig) -> [config.fingerprint_dim]: WINDOW-FREE ONLY. FORBIDDEN: any window like [-5700:-500], any TTC, any TimeWindowConfig. No GatingWindowConfig dataclass. Exact 16-dim:
      [0] mean gate_lif, [1] std, [2] max, [3] min, [4] AUC=mean, [5] entropy (hist config.entropy_bins bins [0,1] eps=1e-12)
      [6-11] same 6 for gate_gru
      [12] Pearson corr (0.0 if std==0 else corr, NaN->0.0)
      [13] time max gate_lif / T, [14] time min / T, [15] max |grad(gate_lif)|
      All use config.fingerprint_dim, config.entropy_bins, masking for variable T. Interpolation to config.interp_length only for viz.

    - def cluster(fingerprints[N,16], config): StandardScaler, silhouette for k in config.n_clusters_range list [2,3,4,5] without labels -> k_opt, then KMeans/GMM for k=4 and k=3. All seeded: np.random.seed, random.seed, torch.manual_seed(config.random_state), KMeans(random_state=config.random_state, n_init=10), GMM(random_state), UMAP(random_state, n_jobs=1).

2. Config dataclasses:
   @dataclass(frozen=True) class ClusterGatingConfig:
   n_clusters: int=4, n_clusters_range: list[int]=field(default_factory=lambda: [2,3,4,5]), random_state:int=42, use_umap:bool=True, fingerprint_dim:int=16, entropy_bins:int=20, interp_length:int=200
   No dict, no window field. Register in config_parser.py. Add to config/default.yaml same values, no window key.

3. Dependencies: Add umap-learn to requirements.txt AND pyproject.toml AND Dockerfile.

4. scripts/analyze_gating.py + Makefile: create cluster: target, add to analyze: deps (now 6). Save to results/ 300 DPI: gating_umap_true_4way.png, gating_umap_true_3way.png, gating_umap_pred_k4.png, gating_umap_pred_kopt.png, gating_trajectories_by_cluster.png (x normalized 0-1), gating_cluster_summary.json {k_opt, silhouette_scores, ARI_4way_k4, NMI_4way_k4, ARI_3way_k3, NMI_3way_k3, window_used:false, config_hash via json.dumps sort_keys, mapping_definition}, gating_cluster_statistics.csv. Ensure make pipeline, make test, docker compose run --rm nsmor analyze/cluster work.

5. tests/test_gating_cluster.py: mock gates [4,100,2], fingerprint shape [4,16], entropy deterministic, Pearson NaN guard, no forbidden strings ("5700","TTC","baseline_duration","TimeWindowConfig") outside docstring, determinism, mapping 4->3. Must pass make test.

6. Docstring verbatim: "Window-free by design. NSMoR is Trial-Start anchored. TTC-50ms is only for MCMC prior 5-D snapshot. Baseline 5700ms is variant for pure-wind via TimeWindowConfig, not universal. Manual windows like [-5700:-500] inject human bias and break unsupervised claim. Clustering is unsupervised (silhouette selects k without labels); k=4 matches labeling.py cardinality; k=3 merged is for biological interpretation only and defined as Startle->Escape, Walk+Pre_Active->PreWalk, NoResponse->NoResponse. Pearson NaN guarded to 0.0 when std==0."

Do NOT modify model, loss, train. Keep RNG handling from checkpoint.py.
