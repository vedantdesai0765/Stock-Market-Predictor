import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

const FIVE_MINUTES = 5 * 60 * 1000;

/** Today's Morning Call. Refreshes by itself so an open tab picks up the forecast and its results. */
export function useMorningCall() {
  return useQuery({
    queryKey: ["morning-call"],
    queryFn: () => api.getMorningCall(),
    refetchInterval: FIVE_MINUTES,
    refetchOnWindowFocus: true,
    staleTime: 60 * 1000,
  });
}

/** Live track record of the Morning Call, plus the walk-forward test results for comparison. */
export function useMorningScorecard(recentDays: number = 5) {
  return useQuery({
    queryKey: ["morning-scorecard", recentDays],
    queryFn: () => api.getMorningScorecard(recentDays),
    refetchInterval: FIVE_MINUTES,
    refetchOnWindowFocus: true,
    staleTime: 60 * 1000,
  });
}
