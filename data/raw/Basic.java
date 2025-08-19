package data.raw;

public class Basic {
    
    public static void main(String[] args) {
        System.out.println("This is a basic Java program.");

        byte age = 25;

        // Conditional example
        if (age >= 18) {
            System.out.println("You are an adult.");
        } else {
            System.out.println("You are a minor.");
        }

        // Loop example
        for (int i = 0; i < 5; i++) {
            System.out.println("Count: " + i);
        }
    }
}
