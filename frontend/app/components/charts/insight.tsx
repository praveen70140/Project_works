import {
  ratingApi,
  successProbabilityApi,
  unifiedApi,
} from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { useFormContext } from "react-hook-form";
import TrendCard from "../ui/trend-card";
import { Icon, UserIcon } from "lucide-react";
import { DynamicIcon } from "lucide-react/dynamic";
import { months } from "@/app/utils/time";

interface CardProps {
  iconName: string;
  label: string;
  value: string | number;
  width: 2 | 3;
}

const Insight = ({ data }: { data: unifiedApi }) => {
  const { gemini_recommendation } = data;

  return (
    <Card className="col-span-6">
      <CardBody className="flex flex-col justify-center items-center">
        <h3 className="self-start">Insight</h3>
        <p className="text-xl text-justify text-default-600">
          {gemini_recommendation}
        </p>
      </CardBody>
    </Card>
  );
};

export default Insight;
