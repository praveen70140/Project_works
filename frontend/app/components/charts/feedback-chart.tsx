"use client";
import { unifiedApi } from "@/app/types/api-types";
import { capitalize } from "@/app/utils/text";
import { Card, CardBody } from "@heroui/react";
import { StarIcon } from "lucide-react";

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

const FeedbackChart = ({ data }: { data: unifiedApi }) => {
  const feedbackState = data.feedback_prediction.feedback_prediction;

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
      <CardBody className="flex items-center">
        <h3 className="self-start justify-self-start">Rating Prediction</h3>
        <div className="my-auto">
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
        </div>
      </CardBody>
    </Card>
  );
};

export default FeedbackChart;
