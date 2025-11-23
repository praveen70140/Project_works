import {
  ratingApi,
  successProbabilityApi,
  unifiedApi,
} from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { useFormContext } from "react-hook-form";
import useSWR from "swr";

const RatingChart = ({ data }: { data: unifiedApi }) => {
  const { rf_rating_prediction } = data.rf_rating_prediction;

  return (
    <Card className="col-span-2">
      <CardBody className="flex flex-col items-center">
        <h3 className="self-start">Expected Rating</h3>

        <CircularProgress
          classNames={{
            svg: "w-36 h-36 drop-shadow-md",
            indicator: "",
            track: "",
            value: "text-3xl font-semibold ",
          }}
          color="warning"
          minValue={0}
          maxValue={5}
          showValueLabel={true}
          formatOptions={{ style: "decimal" }}
          label="Stars"
          strokeWidth={4}
          value={Number(rf_rating_prediction.toFixed(2))}
        />
      </CardBody>
    </Card>
  );
};

export default RatingChart;
