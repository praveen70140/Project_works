"use client";
import { Controller, useFormContext } from "react-hook-form";
import { mainFormData } from "../types/main-form-data";
import {
  Button,
  DatePicker,
  Input,
  Select,
  SelectItem,
  Slider,
} from "@heroui/react";
import citiesList from "@/app/data/cities.json";
import cuisineList from "@/app/data/cuisines.json";
import { CalendarIcon } from "lucide-react";

interface Props {
  onSubmit: () => void;
}

const MainForm = ({ onSubmit }: Props) => {
  const { handleSubmit, register, control } = useFormContext<mainFormData>();

  return (
    <div className="">
      <form
        onSubmit={handleSubmit(onSubmit)}
        className="flex flex-col space-y-4 p-4"
      >
        <div>
          <h3 className="text-xl font-medium">Enter the following details</h3>
          <p className="text-default-500">
            We will use this to give a prediction model of your business
          </p>
        </div>
        <div className="flex space-x-2">
          <Input
            type="text"
            label="Restaurant Name"
            {...register("restaurantName", { required: true })}
          />
          <Select
            isVirtualized
            label={"Cuisine"}
            placeholder="Select..."
            {...register("cuisine", { required: true })}
          >
            {cuisineList.map((cuisine: string) => (
              <SelectItem key={cuisine}>{cuisine}</SelectItem>
            ))}
          </Select>
        </div>
        <div className="flex space-x-2">
          <Input
            type="text"
            label="Location"
            {...register("location", { required: true })}
          />

          <Select
            isVirtualized
            label={"City"}
            placeholder="Select..."
            {...register("city", { required: true })}
          >
            {citiesList.map((city: string) => (
              <SelectItem key={city}>{city}</SelectItem>
            ))}
          </Select>
        </div>

        <div className="flex space-x-2">
          <Controller
            control={control}
            name="salesAmount"
            rules={{ required: true }}
            render={({ field }) => (
              <Slider
                defaultValue={5000}
                label="Sales Amount"
                maxValue={10000}
                minValue={0}
                step={100}
                {...field}
              />
            )}
          />

          <Controller
            control={control}
            name="salesQuantity"
            rules={{ required: true }}
            render={({ field }) => (
              <Slider
                defaultValue={5000}
                label="Sales Quantity"
                maxValue={10000}
                minValue={0}
                step={100}
                {...field}
              />
            )}
          />

          <Controller
            name="date"
            control={control}
            rules={{ required: true }}
            render={({ field }) => (
              <DatePicker
                endContent={<CalendarIcon />}
                label="Date of establishment"
                {...field}
              />
            )}
          />

          <Controller
            control={control}
            name="rating"
            rules={{ required: true }}
            render={({ field }) => (
              <Slider
                defaultValue={5000}
                label="Rating"
                maxValue={5}
                minValue={1}
                step={0.1}
                {...field}
              />
            )}
          />
        </div>
        <Button className="mt-3 w-10" size="lg" type="submit" color="primary">
          Submit
        </Button>
      </form>
    </div>
  );
};

export default MainForm;
