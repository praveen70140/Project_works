import { ratingApi, successProbabilityApi } from "@/app/types/api-types";
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

const UserDetailsCard = ({ iconName, label, value, width }: CardProps) => {
  const widthClass =
    width === 2 ? "col-span-2" : width === 3 ? "col-span-3" : "";
  return (
    <Card className={widthClass}>
      <CardBody className="flex flex-row items-center space-x-3">
        <div>
          <DynamicIcon color="grey" name={iconName as any} size={36} />
        </div>
        <div>
          <p className="text-sm font-semibold">{label}</p>
          <p className="text-2xl">{value}</p>
        </div>
      </CardBody>
    </Card>
  );
};

const UserDetails = () => {
  const { getValues } = useFormContext<mainFormData>();
  const payload = getValues();

  return (
    <Card className="col-span-6">
      <CardBody className="flex flex-col items-center">
        <h3 className="self-start">User Details</h3>
        <div className="grid grid-cols-6 w-full gap-2">
          <UserDetailsCard
            iconName="landmark"
            width={3}
            label="Restaurant Name"
            value={payload.restaurantName}
          />

          <UserDetailsCard
            width={3}
            iconName="utensils-crossed"
            label="Cuisine"
            value={payload.cuisine}
          />
          <UserDetailsCard
            iconName="package"
            label="Expected Quantity"
            value={payload.salesQuantity}
            width={2}
          />
          <UserDetailsCard
            iconName="circle-dollar-sign"
            label="Expected Sales"
            value={payload.salesAmount}
            width={2}
          />
          <UserDetailsCard
            iconName="star"
            label="Rating"
            value={payload.rating}
            width={2}
          />
          <UserDetailsCard
            iconName="calendar-days"
            label="Month"
            value={months[payload.date.month]}
            width={2}
          />
          <UserDetailsCard
            iconName="calendar"
            label="Year"
            value={payload.date.year}
            width={2}
          />
          <UserDetailsCard
            iconName="building-2"
            label="City"
            value={payload.city}
            width={2}
          />
        </div>
      </CardBody>
    </Card>
  );
};

export default UserDetails;
