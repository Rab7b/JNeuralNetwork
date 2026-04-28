import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import lib.ai.network.Model;
import lib.ai.neuron.Savings;
import lib.plugins.Mnist;

public class Main extends JFrame {
    private Model network;
    private BufferedImage canvas;
    private Graphics2D g2;
    private JLabel labelResult;

    public Main() throws Exception{

        network = new Model(new int[]{784, 64, 32, 10}, 0);
        Savings loader = new Savings("weights.txt");
        network.setWeightsFlat(loader.loadMain());

        setTitle("MNIST Digit Recognizer");
        setSize(400, 450);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        canvas = new BufferedImage(280, 280, BufferedImage.TYPE_INT_RGB);
        g2 = canvas.createGraphics();
        g2.setColor(Color.BLACK);
        g2.fillRect(0, 0, 280, 280);
        g2.setStroke(new BasicStroke(20));
        g2.setColor(Color.WHITE);

        JPanel drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                g.drawImage(canvas, 0, 0, null);
            }
        };

        drawPanel.addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(MouseEvent e) {
                g2.fillOval(e.getX(), e.getY(), 20, 20); 
                drawPanel.repaint();
            }
        });

        labelResult = new JLabel("Draw a digit!", SwingConstants.CENTER);
        labelResult.setFont(new Font("Arial", Font.BOLD, 24));

        JButton btnClear = new JButton("Clear");
        btnClear.addActionListener(e -> {
            g2.setColor(Color.BLACK);
            g2.fillRect(0, 0, 280, 280);
            g2.setColor(Color.WHITE);
            labelResult.setText("Cleared!");
            drawPanel.repaint();
        });

        JButton btnGuess = new JButton("Guess!");
        btnGuess.addActionListener(e -> guessDigit());

        JPanel bottomPanel = new JPanel(new GridLayout(1, 2));
        bottomPanel.add(btnClear);
        bottomPanel.add(btnGuess);

        add(drawPanel, BorderLayout.CENTER);
        add(labelResult, BorderLayout.NORTH);
        add(bottomPanel, BorderLayout.SOUTH);

        setVisible(true);
    }

    private void guessDigit() {

        Image scaled = canvas.getScaledInstance(28, 28, Image.SCALE_SMOOTH);
        BufferedImage smallImg = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = smallImg.createGraphics();
        g.drawImage(scaled, 0, 0, null);
        g.dispose();

        double[] inputs = new double[784];
        int n = 0;
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = smallImg.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                inputs[n++] = r / 255.0; 
            }
        }

        double[] output = network.predict(inputs);
        int guess = getArgMax(output);
        labelResult.setText("I think it is: " + guess);
    }

    private int getArgMax(double[] array) {
        int max = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[max]) max = i;
        }
        return max;
    }

    public static void main(String[] args) throws Exception{
        Mnist m = new Mnist();
        m.train();
        new Main();
    }
}