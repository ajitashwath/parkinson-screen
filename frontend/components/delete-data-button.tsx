"use client";

import { useState } from "react";
import { removeLocalData } from "@/lib/client-data";
import { deleteSessionsFromApi } from "@/lib/api";

export function DeleteDataButton() {
  const [deleted, setDeleted] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDelete = async () => {
    setIsDeleting(true);
    setError(null);
    try {
      await deleteSessionsFromApi();
      removeLocalData();
      setDeleted(true);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "The data deletion request failed.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        className="rounded-[4px] border border-violet px-4 py-2 font-mono text-[10px] uppercase tracking-[0.1em] text-violet transition-colors hover:bg-linen disabled:cursor-not-allowed disabled:opacity-50"
        disabled={deleted || isDeleting}
        onClick={() => void handleDelete()}
        type="button"
      >
        {deleted ? "Data deleted" : isDeleting ? "Deleting" : "Delete my data"}
      </button>
      {deleted ? <span className="text-xs text-abyss">Your browser history was cleared.</span> : null}
      {error ? <span className="text-xs text-violet" role="alert">{error}</span> : null}
    </div>
  );
}
