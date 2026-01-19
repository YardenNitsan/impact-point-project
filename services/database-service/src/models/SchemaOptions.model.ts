import { SchemaOptions } from "mongoose";

export const defaultSchemaOptions: SchemaOptions = {
    timestamps: {
        createdAt: true,
        updatedAt: false
    },
    versionKey: false
}