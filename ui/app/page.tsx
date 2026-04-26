"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type ScreenResponse = {
  cand_id: string;
  decision: "advance" | "reject";
  score: number;
  fairness_postprocessed_decision: "advance" | "reject";
  audit: Record<string, unknown>;
  disclaimer: string;
};

type AuditRow = {
  group: string;
  n: number;
  recall: number;
  fpr: number;
  selection_rate: number;
};

type AuditBlock = {
  rows: AuditRow[];
  recall_gap: number;
  selection_rate_gap: number;
  fpr_gap: number;
};

type AuditResponse = {
  baseline?: AuditBlock;
  fairlearn?: AuditBlock;
  note?: string;
  fairlearn_error?: string;
};

const DEMO = {
  cand_id: "C-DEMO-001",
  years_experience: 6,
  education_level: "Bachelor" as const,
  gender: "Female" as const,
  nationality_group: "South Asian" as const,
  prior_employer_tier: 2,
  skill_tfidf_features: Array.from({ length: 32 }, (_, i) => (i % 5 === 0 ? 0.6 : 0.1)),
};

export default function Home() {
  const [pred, setPred] = useState<ScreenResponse | null>(null);
  const [audit, setAudit] = useState<AuditResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [auditLoading, setAuditLoading] = useState(false);

  async function score() {
    setLoading(true);
    const res = await fetch(`${API}/screen`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(DEMO),
    });
    setPred(await res.json());
    setLoading(false);
  }

  async function loadAudit() {
    setAuditLoading(true);
    const res = await fetch(`${API}/audit?sensitive=gender`);
    setAudit(await res.json());
    setAuditLoading(false);
  }

  useEffect(() => {
    loadAudit();
  }, []);

  return (
    <main className="min-h-screen p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold">Fair AI Hiring</h1>
      <p className="opacity-70 mb-6">
        Résumé screening with a baseline + a Fairlearn (equalized-odds) post-processed decision, side by side.
      </p>

      <button
        onClick={score}
        disabled={loading}
        className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
      >
        {loading ? "Scoring..." : "Score demo candidate"}
      </button>

      {pred && (
        <section className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Stat label="Score" value={(pred.score * 100).toFixed(1) + "%"} />
          <Stat
            label="Baseline decision"
            value={pred.decision}
            color={pred.decision === "advance" ? "text-green-600" : "text-red-600"}
          />
          <Stat
            label="Fairness-postprocessed"
            value={pred.fairness_postprocessed_decision}
            color={
              pred.fairness_postprocessed_decision === "advance"
                ? "text-green-600"
                : "text-red-600"
            }
          />
        </section>
      )}
      {pred && <p className="mt-3 text-xs opacity-60 italic">{pred.disclaimer}</p>}

      <section className="mt-10">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-semibold">Subgroup audit (gender)</h2>
          <button
            onClick={loadAudit}
            disabled={auditLoading}
            className="text-sm rounded-xl px-3 py-1 border"
          >
            {auditLoading ? "Refreshing..." : "Refresh audit"}
          </button>
        </div>

        {audit?.note && <p className="opacity-70 text-sm">{audit.note}</p>}
        {audit?.baseline && <AuditTable title="Baseline" block={audit.baseline} />}
        {audit?.fairlearn && (
          <div className="mt-6">
            <AuditTable title="Fairlearn (Equalized Odds)" block={audit.fairlearn} />
          </div>
        )}
        {audit?.fairlearn_error && (
          <p className="mt-3 text-xs text-red-600">Fairlearn audit error: {audit.fairlearn_error}</p>
        )}
      </section>
    </main>
  );
}

function Stat({
  label,
  value,
  color = "",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="rounded-2xl border p-4">
      <div className="text-xs uppercase tracking-wide opacity-60">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${color}`}>{value}</div>
    </div>
  );
}

function AuditTable({ title, block }: { title: string; block: AuditBlock }) {
  const fail = block.recall_gap > 0.05 || block.selection_rate_gap > 0.05 || block.fpr_gap > 0.05;
  return (
    <div className="rounded-2xl border p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold">{title}</div>
        <div
          className={
            "text-xs px-2 py-1 rounded-full " +
            (fail ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700")
          }
        >
          {fail ? "audit FAIL — gap > 5pts" : "audit pass"}
        </div>
      </div>
      <table className="w-full text-sm mt-3">
        <thead>
          <tr className="text-left opacity-60">
            <th>group</th>
            <th>n</th>
            <th>recall</th>
            <th>FPR</th>
            <th>selection rate</th>
          </tr>
        </thead>
        <tbody>
          {block.rows.map((r) => (
            <tr key={r.group} className="border-t">
              <td className="py-2">{r.group}</td>
              <td>{r.n}</td>
              <td>{r.recall.toFixed(3)}</td>
              <td>{r.fpr.toFixed(3)}</td>
              <td>{r.selection_rate.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 text-xs opacity-60">
        gaps — recall: {block.recall_gap.toFixed(3)} · FPR: {block.fpr_gap.toFixed(3)} · selection rate:{" "}
        {block.selection_rate_gap.toFixed(3)}
      </div>
    </div>
  );
}
