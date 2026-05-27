import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  CheckCircle,
  XCircle,
  Clock,
  Loader,
  Send,
  Eye,
  BarChart2,
  Zap,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Minimal Markdown → HTML converter (no external deps)
// ---------------------------------------------------------------------------
function markdownToHtml(md) {
  if (!md) return "";
  let html = md
    // Escape existing HTML entities first to prevent injection
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Fenced code blocks
  html = html.replace(/```[\w]*\n([\s\S]*?)```/g, (_, code) => {
    return `<pre class="bg-gray-950 rounded p-3 my-3 overflow-x-auto text-sm text-green-400 font-mono"><code>${code.trim()}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-gray-950 text-green-400 font-mono px-1 rounded text-sm">$1</code>');

  // Headings
  html = html.replace(/^#{6}\s+(.+)$/gm, '<h6 class="text-xs font-bold mt-3 mb-1 text-gray-100">$1</h6>');
  html = html.replace(/^#{5}\s+(.+)$/gm, '<h5 class="text-sm font-bold mt-3 mb-1 text-gray-100">$1</h5>');
  html = html.replace(/^#{4}\s+(.+)$/gm, '<h4 class="text-base font-bold mt-4 mb-1 text-gray-100">$1</h4>');
  html = html.replace(/^#{3}\s+(.+)$/gm, '<h3 class="text-lg font-bold mt-4 mb-2 text-gray-100">$1</h3>');
  html = html.replace(/^#{2}\s+(.+)$/gm, '<h2 class="text-xl font-bold mt-5 mb-2 text-gray-100">$1</h2>');
  html = html.replace(/^#{1}\s+(.+)$/gm, '<h1 class="text-2xl font-bold mt-5 mb-2 text-white">$1</h1>');

  // Bold + Italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="font-bold">$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong class="font-bold">$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em class="italic">$1</em>');
  html = html.replace(/_(.+?)_/g, '<em class="italic">$1</em>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-400 underline" target="_blank" rel="noopener noreferrer">$1</a>');

  // Unordered lists
  html = html.replace(/^[-*+]\s+(.+)$/gm, '<li class="ml-4 list-disc">$1</li>');
  html = html.replace(/(<li[\s\S]*?<\/li>)/g, (block) => {
    if (!block.startsWith('<ul')) return `<ul class="my-2 space-y-1">${block}</ul>`;
    return block;
  });

  // Ordered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li class="ml-4 list-decimal">$1</li>');

  // Horizontal rules
  html = html.replace(/^---+$/gm, '<hr class="border-gray-600 my-4" />');

  // Blockquotes
  html = html.replace(/^>\s+(.+)$/gm, '<blockquote class="border-l-4 border-gray-500 pl-4 italic text-gray-400 my-2">$1</blockquote>');

  // Paragraphs — wrap non-tag lines separated by blank lines
  const lines = html.split("\n");
  const result = [];
  let inParagraph = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isTagLine = /^\s*<(h[1-6]|ul|ol|li|pre|hr|blockquote|p)/.test(line);
    if (line.trim() === "") {
      if (inParagraph) {
        result.push("</p>");
        inParagraph = false;
      }
    } else if (isTagLine) {
      if (inParagraph) {
        result.push("</p>");
        inParagraph = false;
      }
      result.push(line);
    } else {
      if (!inParagraph) {
        result.push('<p class="my-2 text-gray-300 leading-relaxed">');
        inParagraph = true;
      }
      result.push(line);
    }
  }
  if (inParagraph) result.push("</p>");

  return result.join("\n");
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const BASE_URL = "http://localhost:8000";

const STATUS_CONFIG = {
  running: {
    label: "Running",
    color: "text-blue-400",
    bg: "bg-blue-900/30 border-blue-700",
    Icon: Loader,
    spin: true,
  },
  waiting_approval: {
    label: "Awaiting Approval",
    color: "text-yellow-400",
    bg: "bg-yellow-900/30 border-yellow-700",
    Icon: Clock,
    spin: false,
  },
  completed: {
    label: "Completed",
    color: "text-green-400",
    bg: "bg-green-900/30 border-green-700",
    Icon: CheckCircle,
    spin: false,
  },
  rejected: {
    label: "Rejected",
    color: "text-red-400",
    bg: "bg-red-900/30 border-red-700",
    Icon: XCircle,
    spin: false,
  },
  error: {
    label: "Error",
    color: "text-red-500",
    bg: "bg-red-900/40 border-red-600",
    Icon: XCircle,
    spin: false,
  },
};

const PIPELINE_NODES = [
  "ingestor",
  "planner",
  "researcher",
  "writer",
  "reviewer",
  "optimizer",
  "hitl",
];

// ---------------------------------------------------------------------------
// Helper components
// ---------------------------------------------------------------------------
function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.running;
  const { label, color, bg, Icon, spin } = cfg;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${color} ${bg}`}
    >
      <Icon size={12} className={spin ? "animate-spin" : ""} />
      {label}
    </span>
  );
}

function SectionCard({ title, icon: Icon, children, className = "" }) {
  return (
    <div className={`bg-gray-800 border border-gray-700 rounded-xl shadow-lg ${className}`}>
      {title && (
        <div className="flex items-center gap-2 px-5 py-4 border-b border-gray-700">
          {Icon && <Icon size={16} className="text-indigo-400" />}
          <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wider">
            {title}
          </h2>
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}

function InputField({ label, type = "text", value, onChange, placeholder, className = "" }) {
  return (
    <div className={`flex flex-col gap-1.5 ${className}`}>
      {label && (
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {label}
        </label>
      )}
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
      />
    </div>
  );
}

function SelectField({ label, value, onChange, options, className = "" }) {
  return (
    <div className={`flex flex-col gap-1.5 ${className}`}>
      {label && (
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {label}
        </label>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors appearance-none cursor-pointer"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function Button({ children, onClick, variant = "primary", disabled = false, className = "" }) {
  const variants = {
    primary: "bg-indigo-600 hover:bg-indigo-500 text-white border-indigo-500",
    success: "bg-green-600 hover:bg-green-500 text-white border-green-500",
    danger: "bg-red-600 hover:bg-red-500 text-white border-red-500",
    ghost: "bg-transparent hover:bg-gray-700 text-gray-300 border-gray-600",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${variants[variant]} ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"} ${className}`}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Auth Screen
// ---------------------------------------------------------------------------
function AuthScreen({ onAuth }) {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = useCallback(async () => {
    if (!username.trim()) {
      setError("Username is required.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${BASE_URL}/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim() }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const token = data.access_token || data.token || data;
      if (!token || typeof token !== "string") {
        throw new Error("Invalid token response from server.");
      }
      onAuth(token);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [username, onAuth]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter") handleLogin();
    },
    [handleLogin]
  );

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-indigo-600 rounded-2xl mb-4 shadow-lg shadow-indigo-900/50">
            <Zap size={28} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">AI Content Engine</h1>
          <p className="text-gray-400 text-sm mt-1">Human-in-the-Loop Approval Dashboard</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-2xl p-8 shadow-xl">
          <h2 className="text-lg font-semibold text-gray-100 mb-6">Sign In</h2>
          <div className="space-y-4">
            <InputField
              label="Username"
              value={username}
              onChange={setUsername}
              placeholder="Enter your username"
              onKeyDown={handleKeyDown}
            />
            <div
              onKeyDown={handleKeyDown}
              className="hidden"
            />
          </div>
          {error && (
            <div className="mt-4 flex items-start gap-2 bg-red-900/30 border border-red-700 rounded-lg p-3 text-sm text-red-400">
              <XCircle size={16} className="shrink-0 mt-0.5" />
              {error}
            </div>
          )}
          <Button
            variant="primary"
            onClick={handleLogin}
            disabled={loading}
            className="w-full mt-6 justify-center"
          >
            {loading ? (
              <>
                <Loader size={16} className="animate-spin" /> Authenticating…
              </>
            ) : (
              <>
                <Send size={16} /> Get Access Token
              </>
            )}
          </Button>
          <p className="text-xs text-gray-500 mt-4 text-center">
            Token is kept in memory only and never persisted to disk.
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// New Run Panel
// ---------------------------------------------------------------------------
function NewRunPanel({ token, onRunCreated }) {
  const [topic, setTopic] = useState("");
  const [contentType, setContentType] = useState("blog_post");
  const [platform, setPlatform] = useState("wordpress");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const contentTypeOptions = [
    { value: "blog_post", label: "Blog Post" },
    { value: "newsletter", label: "Newsletter" },
    { value: "social_thread", label: "Social Thread" },
    { value: "product_description", label: "Product Description" },
  ];

  const platformOptions = [
    { value: "wordpress", label: "WordPress" },
    { value: "linkedin", label: "LinkedIn" },
    { value: "twitter", label: "Twitter / X" },
  ];

  const handleSubmit = useCallback(async () => {
    if (!topic.trim()) {
      setError("Topic is required.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${BASE_URL}/runs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          topic: topic.trim(),
          content_type: contentType,
          platform,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const run = await res.json();
      setTopic("");
      onRunCreated(run);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [topic, contentType, platform, token, onRunCreated]);

  return (
    <SectionCard title="New Content Run" icon={Zap}>
      <div className="space-y-4">
        <InputField
          label="Topic"
          value={topic}
          onChange={setTopic}
          placeholder="e.g. The future of AI in healthcare"
        />
        <div className="grid grid-cols-2 gap-4">
          <SelectField
            label="Content Type"
            value={contentType}
            onChange={setContentType}
            options={contentTypeOptions}
          />
          <SelectField
            label="Platform"
            value={platform}
            onChange={setPlatform}
            options={platformOptions}
          />
        </div>
        {error && (
          <div className="flex items-start gap-2 bg-red-900/30 border border-red-700 rounded-lg p-3 text-sm text-red-400">
            <XCircle size={16} className="shrink-0 mt-0.5" />
            {error}
          </div>
        )}
        <Button
          variant="primary"
          onClick={handleSubmit}
          disabled={loading}
          className="w-full justify-center"
        >
          {loading ? (
            <>
              <Loader size={16} className="animate-spin" /> Starting Run…
            </>
          ) : (
            <>
              <Send size={16} /> Start Run
            </>
          )}
        </Button>
      </div>
    </SectionCard>
  );
}

// ---------------------------------------------------------------------------
// Run List Item
// ---------------------------------------------------------------------------
function RunListItem({ run, isSelected, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border transition-all ${
        isSelected
          ? "bg-indigo-900/40 border-indigo-600"
          : "bg-gray-900/50 border-gray-700 hover:border-gray-500 hover:bg-gray-700/30"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs font-mono text-gray-400 truncate">{run.thread_id}</p>
          {run.topic && (
            <p className="text-sm text-gray-200 mt-0.5 truncate font-medium">{run.topic}</p>
          )}
          {run.content_type && (
            <p className="text-xs text-gray-500 mt-0.5 capitalize">
              {run.content_type.replace(/_/g, " ")} · {run.platform}
            </p>
          )}
        </div>
        <StatusBadge status={run.status} />
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Pipeline Timeline
// ---------------------------------------------------------------------------
function PipelineTimeline({ completedNodes = [], currentNode = null }) {
  return (
    <div className="flex flex-wrap gap-2">
      {PIPELINE_NODES.map((node, idx) => {
        const isDone = completedNodes.includes(node);
        const isCurrent = currentNode === node;
        const isPending = !isDone && !isCurrent;
        return (
          <div key={node} className="flex items-center gap-1">
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium transition-all ${
                isDone
                  ? "bg-green-900/40 border-green-600 text-green-400"
                  : isCurrent
                  ? "bg-indigo-900/40 border-indigo-500 text-indigo-300"
                  : "bg-gray-900/50 border-gray-700 text-gray-500"
              }`}
            >
              {isDone ? (
                <CheckCircle size={11} />
              ) : isCurrent ? (
                <Loader size={11} className="animate-spin" />
              ) : (
                <Clock size={11} />
              )}
              {node}
            </div>
            {idx < PIPELINE_NODES.length - 1 && (
              <div className={`w-4 h-px ${isDone ? "bg-green-700" : "bg-gray-700"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live Stream Panel
// ---------------------------------------------------------------------------
function LiveStreamPanel({ token, run }) {
  const [streamText, setStreamText] = useState("");
  const [tokenCount, setTokenCount] = useState(0);
  const [estimatedCost, setEstimatedCost] = useState(0);
  const [completedNodes, setCompletedNodes] = useState([]);
  const [currentNode, setCurrentNode] = useState(null);
  const [connected, setConnected] = useState(false);
  const [streamError, setStreamError] = useState("");
  const streamEndRef = useRef(null);
  const esRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    streamEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [streamText, scrollToBottom]);

  useEffect(() => {
    if (!run?.thread_id || !token) return;

    // Cleanup previous connection
    if (esRef.current) {
      esRef.current.close();
    }
    setStreamText("");
    setTokenCount(0);
    setEstimatedCost(0);
    setCompletedNodes([]);
    setCurrentNode(null);
    setStreamError("");
    setConnected(false);

    const url = `${BASE_URL}/runs/${run.thread_id}/stream`;
    const es = new EventSource(`${url}?token=${encodeURIComponent(token)}`);
    esRef.current = es;

    es.onopen = () => setConnected(true);

    es.addEventListener("token", (e) => {
      try {
        const data = JSON.parse(e.data);
        setStreamText((prev) => prev + (data.text || data.content || data || ""));
        setTokenCount((prev) => prev + 1);
        // Rough cost estimate: ~$0.000003 per token (GPT-4o rate)
        setEstimatedCost((prev) => +(prev + 0.000003).toFixed(6));
      } catch {
        setStreamText((prev) => prev + (e.data || ""));
        setTokenCount((prev) => prev + 1);
        setEstimatedCost((prev) => +(prev + 0.000003).toFixed(6));
      }
    });

    es.addEventListener("node_complete", (e) => {
      try {
        const data = JSON.parse(e.data);
        const nodeName = data.node || data.name || data;
        if (typeof nodeName === "string") {
          setCompletedNodes((prev) =>
            prev.includes(nodeName) ? prev : [...prev, nodeName]
          );
          setCurrentNode(null);
        }
      } catch {}
    });

    es.addEventListener("node_start", (e) => {
      try {
        const data = JSON.parse(e.data);
        const nodeName = data.node || data.name || data;
        if (typeof nodeName === "string") setCurrentNode(nodeName);
      } catch {}
    });

    es.addEventListener("error_event", (e) => {
      try {
        const data = JSON.parse(e.data);
        setStreamError(data.message || data.detail || "Stream error");
      } catch {
        setStreamError("Stream connection error");
      }
    });

    es.onerror = () => {
      setConnected(false);
      // Only set error if we haven't received a clean close
      es.close();
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [run?.thread_id, token]);

  const costColor =
    estimatedCost < 0.01
      ? "text-green-400"
      : estimatedCost < 0.05
      ? "text-yellow-400"
      : "text-red-400";

  return (
    <div className="space-y-4">
      {/* Pipeline Progress */}
      <div>
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
          Pipeline Progress
        </p>
        <PipelineTimeline completedNodes={completedNodes} currentNode={currentNode} />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3 text-center">
          <div className="flex items-center justify-center gap-1 text-indigo-400 mb-1">
            <BarChart2 size={14} />
            <span className="text-xs font-medium uppercase tracking-wider">Tokens</span>
          </div>
          <p className="text-xl font-bold text-white">{tokenCount.toLocaleString()}</p>
        </div>
        <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3 text-center">
          <div className="flex items-center justify-center gap-1 text-indigo-400 mb-1">
            <Zap size={14} />
            <span className="text-xs font-medium uppercase tracking-wider">Est. Cost</span>
          </div>
          <p className={`text-xl font-bold ${costColor}`}>${estimatedCost.toFixed(4)}</p>
        </div>
        <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3 text-center">
          <div className="flex items-center justify-center gap-1 text-indigo-400 mb-1">
            <Eye size={14} />
            <span className="text-xs font-medium uppercase tracking-wider">Status</span>
          </div>
          <p className={`text-sm font-semibold mt-1 ${connected ? "text-green-400" : "text-gray-500"}`}>
            {connected ? "Live" : "Offline"}
          </p>
        </div>
      </div>

      {streamError && (
        <div className="flex items-start gap-2 bg-red-900/30 border border-red-700 rounded-lg p-3 text-sm text-red-400">
          <XCircle size={16} className="shrink-0 mt-0.5" />
          {streamError}
        </div>
      )}

      {/* Live Output */}
      <div>
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
          Live Output
        </p>
        <div className="bg-gray-950 border border-gray-700 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
          {streamText || (
            <span className="text-gray-600 italic">
              {connected ? "Waiting for output…" : "Connecting to stream…"}
            </span>
          )}
          <div ref={streamEndRef} />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// HITL Approval Panel
// ---------------------------------------------------------------------------
function HitlApprovalPanel({ token, run, onResume }) {
  const [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPreview, setShowPreview] = useState(true);

  const hitlData = run?.hitl_data || run?.approval_data || {};
  const {
    title = "",
    meta_description = "",
    seo_score = null,
    slug = "",
    article_body = "",
    body = "",
  } = hitlData;

  const articleContent = article_body || body || "";

  const handleDecision = useCallback(
    async (approved) => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch(`${BASE_URL}/runs/${run.thread_id}/resume`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ approved, feedback: feedback.trim() }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || `HTTP ${res.status}`);
        }
        const updated = await res.json();
        onResume(updated);
        setFeedback("");
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [run?.thread_id, token, feedback, onResume]
  );

  const seoColor =
    seo_score === null
      ? "text-gray-400"
      : seo_score >= 80
      ? "text-green-400"
      : seo_score >= 60
      ? "text-yellow-400"
      : "text-red-400";

  return (
    <div className="space-y-5">
      {/* Header Banner */}
      <div className="flex items-center gap-3 bg-yellow-900/30 border border-yellow-700 rounded-xl p-4">
        <Clock size={20} className="text-yellow-400 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-yellow-300">Awaiting Your Approval</p>
          <p className="text-xs text-yellow-500 mt-0.5">
            Review the generated content below, then approve or reject with feedback.
          </p>
        </div>
      </div>

      {/* SEO Metadata */}
      <div className="grid grid-cols-1 gap-3">
        {title && (
          <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Title</p>
            <p className="text-sm text-gray-100 font-medium">{title}</p>
          </div>
        )}
        {meta_description && (
          <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Meta Description</p>
            <p className="text-sm text-gray-300">{meta_description}</p>
          </div>
        )}
        <div className="grid grid-cols-2 gap-3">
          {slug && (
            <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3">
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Slug</p>
              <p className="text-sm font-mono text-indigo-300">{slug}</p>
            </div>
          )}
          {seo_score !== null && (
            <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">SEO Score</p>
              <p className={`text-2xl font-bold ${seoColor}`}>{seo_score}</p>
              <p className="text-xs text-gray-500">/ 100</p>
            </div>
          )}
        </div>
      </div>

      {/* Article Preview */}
      {articleContent && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Article Preview
            </p>
            <button
              onClick={() => setShowPreview((v) => !v)}
              className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors flex items-center gap-1"
            >
              <Eye size={12} />
              {showPreview ? "Hide" : "Show"}
            </button>
          </div>
          {showPreview && (
            <div className="bg-gray-950 border border-gray-700 rounded-xl p-5 max-h-96 overflow-y-auto prose prose-invert max-w-none text-sm">
              <div
                dangerouslySetInnerHTML={{ __html: markdownToHtml(articleContent) }}
              />
            </div>
          )}
        </div>
      )}

      {/* Feedback */}
      <div>
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider block mb-1.5">
          Feedback (optional)
        </label>
        <textarea
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="Provide feedback for revision, or leave blank to approve as-is…"
          rows={3}
          className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors resize-none"
        />
      </div>

      {error && (
        <div className="flex items-start gap-2 bg-red-900/30 border border-red-700 rounded-lg p-3 text-sm text-red-400">
          <XCircle size={16} className="shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button
          variant="success"
          onClick={() => handleDecision(true)}
          disabled={loading}
          className="flex-1 justify-center"
        >
          {loading ? (
            <Loader size={16} className="animate-spin" />
          ) : (
            <CheckCircle size={16} />
          )}
          Approve & Publish
        </Button>
        <Button
          variant="danger"
          onClick={() => handleDecision(false)}
          disabled={loading}
          className="flex-1 justify-center"
        >
          {loading ? (
            <Loader size={16} className="animate-spin" />
          ) : (
            <XCircle size={16} />
          )}
          Reject
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Run Detail Pane
// ---------------------------------------------------------------------------
function RunDetailPane({ token, run, onRunUpdate }) {
  const showStream =
    run.status === "running" || run.status === "waiting_approval" || run.status === "completed";
  const showHitl = run.status === "waiting_approval";

  return (
    <div className="space-y-5">
      {/* Run Header */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Thread ID</p>
            <p className="font-mono text-sm text-gray-300">{run.thread_id}</p>
            {run.topic && (
              <p className="text-base font-semibold text-white mt-2">{run.topic}</p>
            )}
            {run.content_type && (
              <p className="text-sm text-gray-400 mt-0.5 capitalize">
                {run.content_type.replace(/_/g, " ")}
                {run.platform && ` · ${run.platform}`}
              </p>
            )}
          </div>
          <StatusBadge status={run.status} />
        </div>
        {run.error && (
          <div className="mt-4 flex items-start gap-2 bg-red-900/30 border border-red-700 rounded-lg p-3 text-sm text-red-400">
            <XCircle size={16} className="shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Run Error</p>
              <p className="mt-0.5 text-red-300">{run.error}</p>
            </div>
          </div>
        )}
      </div>

      {/* Live Stream */}
      {showStream && (
        <SectionCard title="Live Stream" icon={Zap}>
          <LiveStreamPanel token={token} run={run} />
        </SectionCard>
      )}

      {/* HITL Panel */}
      {showHitl && (
        <SectionCard title="Content Review" icon={Eye}>
          <HitlApprovalPanel token={token} run={run} onResume={onRunUpdate} />
        </SectionCard>
      )}

      {/* Completed state */}
      {run.status === "completed" && (
        <div className="flex items-center gap-3 bg-green-900/20 border border-green-700 rounded-xl p-4">
          <CheckCircle size={20} className="text-green-400 shrink-0" />
          <div>
            <p className="text-sm font-semibold text-green-300">Content Published</p>
            <p className="text-xs text-green-600 mt-0.5">
              This run completed successfully.
            </p>
          </div>
        </div>
      )}

      {/* Rejected state */}
      {run.status === "rejected" && (
        <div className="flex items-center gap-3 bg-red-900/20 border border-red-700 rounded-xl p-4">
          <XCircle size={20} className="text-red-400 shrink-0" />
          <div>
            <p className="text-sm font-semibold text-red-300">Content Rejected</p>
            <p className="text-xs text-red-600 mt-0.5">
              This run was rejected. Start a new run to try again.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------
function Dashboard({ token, onLogout }) {
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [runsError, setRunsError] = useState("");
  const pollTimers = useRef({});

  // ---------------------------------------------------------------------------
  // Fetch a single run and update it in state
  // ---------------------------------------------------------------------------
  const fetchRun = useCallback(
    async (threadId) => {
      try {
        const res = await fetch(`${BASE_URL}/runs/${threadId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return;
        const run = await res.json();
        setRuns((prev) =>
          prev.map((r) => (r.thread_id === threadId ? { ...r, ...run } : r))
        );
        return run;
      } catch {}
    },
    [token]
  );

  // ---------------------------------------------------------------------------
  // Start / stop polling for a run
  // ---------------------------------------------------------------------------
  const startPolling = useCallback(
    (threadId) => {
      if (pollTimers.current[threadId]) return; // already polling
      const tick = () => {
        fetchRun(threadId).then((run) => {
          if (run && (run.status === "completed" || run.status === "rejected" || run.status === "error")) {
            clearInterval(pollTimers.current[threadId]);
            delete pollTimers.current[threadId];
          }
        });
      };
      pollTimers.current[threadId] = setInterval(tick, 5000);
    },
    [fetchRun]
  );

  // ---------------------------------------------------------------------------
  // Initial run list load
  // ---------------------------------------------------------------------------
  const fetchAllRuns = useCallback(async () => {
    setLoadingRuns(true);
    setRunsError("");
    try {
      const res = await fetch(`${BASE_URL}/runs`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const list = Array.isArray(data) ? data : data.runs || [];
      setRuns(list);
      // Start polling for active runs
      list.forEach((r) => {
        if (r.status === "running" || r.status === "waiting_approval") {
          startPolling(r.thread_id);
        }
      });
    } catch (err) {
      setRunsError(err.message);
    } finally {
      setLoadingRuns(false);
    }
  }, [token, startPolling]);

  useEffect(() => {
    fetchAllRuns();
    return () => {
      // Cleanup all poll timers on unmount
      Object.values(pollTimers.current).forEach(clearInterval);
    };
  }, [fetchAllRuns]);

  // ---------------------------------------------------------------------------
  // Handle new run created
  // ---------------------------------------------------------------------------
  const handleRunCreated = useCallback(
    (run) => {
      setRuns((prev) => {
        const exists = prev.find((r) => r.thread_id === run.thread_id);
        return exists ? prev : [run, ...prev];
      });
      setSelectedRunId(run.thread_id);
      startPolling(run.thread_id);
    },
    [startPolling]
  );

  // ---------------------------------------------------------------------------
  // Handle run update from HITL resume
  // ---------------------------------------------------------------------------
  const handleRunUpdate = useCallback((updatedRun) => {
    setRuns((prev) =>
      prev.map((r) =>
        r.thread_id === updatedRun.thread_id ? { ...r, ...updatedRun } : r
      )
    );
  }, []);

  const selectedRun = runs.find((r) => r.thread_id === selectedRunId) || null;

  const activeRuns = runs.filter(
    (r) => r.status === "running" || r.status === "waiting_approval"
  );
  const finishedRuns = runs.filter(
    (r) => r.status === "completed" || r.status === "rejected" || r.status === "error"
  );

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Top Nav */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow shadow-indigo-900/50">
            <Zap size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white leading-none">AI Content Engine</h1>
            <p className="text-xs text-gray-500 mt-0.5">HITL Dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {activeRuns.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-indigo-300 bg-indigo-900/30 border border-indigo-700 px-2.5 py-1 rounded-full">
              <Loader size={11} className="animate-spin" />
              {activeRuns.length} active {activeRuns.length === 1 ? "run" : "runs"}
            </div>
          )}
          <button
            onClick={onLogout}
            className="text-xs text-gray-400 hover:text-gray-200 transition-colors px-3 py-1.5 rounded-lg hover:bg-gray-700"
          >
            Sign Out
          </button>
        </div>
      </header>

      <div className="flex h-[calc(100vh-65px)]">
        {/* Left Sidebar */}
        <aside className="w-80 shrink-0 bg-gray-850 border-r border-gray-700 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-gray-700">
            <NewRunPanel token={token} onRunCreated={handleRunCreated} />
          </div>

          {/* Run List */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {runsError && (
              <div className="flex items-start gap-2 bg-red-900/30 border border-red-700 rounded-lg p-3 text-xs text-red-400">
                <XCircle size={14} className="shrink-0 mt-0.5" />
                {runsError}
              </div>
            )}

            {loadingRuns && (
              <div className="flex items-center justify-center py-8 gap-2 text-gray-500 text-sm">
                <Loader size={16} className="animate-spin" /> Loading runs…
              </div>
            )}

            {!loadingRuns && runs.length === 0 && (
              <div className="text-center py-10 text-gray-600 text-sm">
                <BarChart2 size={32} className="mx-auto mb-3 opacity-30" />
                No runs yet. Start your first run above.
              </div>
            )}

            {activeRuns.length > 0 && (
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 px-1">
                  Active
                </p>
                <div className="space-y-2">
                  {activeRuns.map((run) => (
                    <RunListItem
                      key={run.thread_id}
                      run={run}
                      isSelected={selectedRunId === run.thread_id}
                      onClick={() => setSelectedRunId(run.thread_id)}
                    />
                  ))}
                </div>
              </div>
            )}

            {finishedRuns.length > 0 && (
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 px-1">
                  History
                </p>
                <div className="space-y-2">
                  {finishedRuns.map((run) => (
                    <RunListItem
                      key={run.thread_id}
                      run={run}
                      isSelected={selectedRunId === run.thread_id}
                      onClick={() => setSelectedRunId(run.thread_id)}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto p-6">
          {selectedRun ? (
            <RunDetailPane
              token={token}
              run={selectedRun}
              onRunUpdate={handleRunUpdate}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-600">
              <Eye size={48} className="mb-4 opacity-20" />
              <p className="text-lg font-medium text-gray-500">No run selected</p>
              <p className="text-sm mt-1 text-gray-600">
                Select a run from the sidebar or start a new one.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root Component
// ---------------------------------------------------------------------------
export default function ContentEngineUI() {
  const [token, setToken] = useState(null);

  const handleAuth = useCallback((jwt) => {
    setToken(jwt);
  }, []);

  const handleLogout = useCallback(() => {
    setToken(null);
  }, []);

  if (!token) {
    return <AuthScreen onAuth={handleAuth} />;
  }

  return <Dashboard token={token} onLogout={handleLogout} />;
}
