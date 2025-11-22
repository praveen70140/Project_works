"use client";
import { feedbackApi } from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { capitalize } from "@/app/utils/text";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { StarIcon } from "lucide-react";
import { useFormContext } from "react-hook-form";
import useSWR from "swr";

enum Colors {
  Green = "#a3e635",
  Yellow = "#facc15",
  Orange = "#fb923c",
}

enum FeedBackStates {
  Poor = "poor feedback",
  Medium = "median feedback",
  Excellent = "excellent feedback",
}

const fetcher = ([url, payload]: [url: string, payload: mainFormData]) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      Resturant_Name: payload.restaurantName,
      Cuisine: payload.cuisine,
      Location: payload.location,
      City: payload.city,
    }),
  }).then((r) => r.json());

const FeedbackChart = () => {
  const { getValues } = useFormContext<mainFormData>();
  const payload = getValues();
  const { data, error, isLoading } = useSWR<feedbackApi>(
    [`${process.env.NEXT_PUBLIC_API_URL}/predict/feedback`, payload],
    fetcher
  );

  if (isLoading)
    return (
      <div>
        <CircularProgress />
      </div>
    );
  if (!data) return <div>Not found</div>;

  const feedbackState = data.feedback_prediction;

  const selectedColor =
    feedbackState === FeedBackStates.Excellent
      ? Colors.Green
      : feedbackState === FeedBackStates.Medium
      ? Colors.Yellow
      : feedbackState === FeedBackStates.Poor
      ? Colors.Orange
      : "black";

  return (
    <Card className="col-span-2">
      <CardBody className="flex items-center justify-center">
        <h3 className="self-start absolute top-3">Rating Prediction</h3>
        <div className="flex space-x-0">
          <StarIcon
            color={selectedColor}
            className="-rotate-6 scale-80 translate-y-2 animate-star-appear"
            fill={selectedColor}
            size={60}
          />
          <StarIcon
            color={
              [FeedBackStates.Excellent, FeedBackStates.Medium].includes(
                feedbackState as FeedBackStates
              )
                ? selectedColor
                : "darkgrey"
            }
            fill={
              [FeedBackStates.Excellent, FeedBackStates.Medium].includes(
                feedbackState as FeedBackStates
              )
                ? selectedColor
                : "darkgrey"
            }
            className="animate-star-appear delay-150"
            size={60}
          />
          <StarIcon
            className="rotate-6 scale-80 translate-y-1 animate-star-appear delay-300"
            color={
              feedbackState === FeedBackStates.Excellent
                ? selectedColor
                : "darkgrey"
            }
            fill={
              feedbackState === FeedBackStates.Excellent
                ? selectedColor
                : "darkgrey"
            }
            size={60}
          />
        </div>
        <p className={`text-2xl`} style={{ color: selectedColor }}>
          {capitalize(feedbackState)}
        </p>
      </CardBody>
    </Card>
  );
};

export default FeedbackChart;
