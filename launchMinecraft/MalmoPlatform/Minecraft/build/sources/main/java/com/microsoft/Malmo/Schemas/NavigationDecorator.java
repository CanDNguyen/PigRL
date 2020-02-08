//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2020.01.14 at 05:11:15 PM PST 
//


package com.microsoft.Malmo.Schemas;

import java.math.BigDecimal;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="randomPlacementProperties" type="{http://ProjectMalmo.microsoft.com}RandomPlacement"/>
 *         &lt;element name="randomizeCompassLocation" type="{http://www.w3.org/2001/XMLSchema}boolean"/>
 *         &lt;element name="minRandomizedDistance" type="{http://www.w3.org/2001/XMLSchema}decimal" minOccurs="0"/>
 *         &lt;element name="maxRandomizedDistance" type="{http://www.w3.org/2001/XMLSchema}decimal" minOccurs="0"/>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {

})
@XmlRootElement(name = "NavigationDecorator")
public class NavigationDecorator {

    @XmlElement(required = true)
    protected RandomPlacement randomPlacementProperties;
    protected boolean randomizeCompassLocation;
    @XmlElement(defaultValue = "0")
    protected BigDecimal minRandomizedDistance;
    @XmlElement(defaultValue = "8")
    protected BigDecimal maxRandomizedDistance;

    /**
     * Gets the value of the randomPlacementProperties property.
     * 
     * @return
     *     possible object is
     *     {@link RandomPlacement }
     *     
     */
    public RandomPlacement getRandomPlacementProperties() {
        return randomPlacementProperties;
    }

    /**
     * Sets the value of the randomPlacementProperties property.
     * 
     * @param value
     *     allowed object is
     *     {@link RandomPlacement }
     *     
     */
    public void setRandomPlacementProperties(RandomPlacement value) {
        this.randomPlacementProperties = value;
    }

    /**
     * Gets the value of the randomizeCompassLocation property.
     * 
     */
    public boolean isRandomizeCompassLocation() {
        return randomizeCompassLocation;
    }

    /**
     * Sets the value of the randomizeCompassLocation property.
     * 
     */
    public void setRandomizeCompassLocation(boolean value) {
        this.randomizeCompassLocation = value;
    }

    /**
     * Gets the value of the minRandomizedDistance property.
     * 
     * @return
     *     possible object is
     *     {@link BigDecimal }
     *     
     */
    public BigDecimal getMinRandomizedDistance() {
        return minRandomizedDistance;
    }

    /**
     * Sets the value of the minRandomizedDistance property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigDecimal }
     *     
     */
    public void setMinRandomizedDistance(BigDecimal value) {
        this.minRandomizedDistance = value;
    }

    /**
     * Gets the value of the maxRandomizedDistance property.
     * 
     * @return
     *     possible object is
     *     {@link BigDecimal }
     *     
     */
    public BigDecimal getMaxRandomizedDistance() {
        return maxRandomizedDistance;
    }

    /**
     * Sets the value of the maxRandomizedDistance property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigDecimal }
     *     
     */
    public void setMaxRandomizedDistance(BigDecimal value) {
        this.maxRandomizedDistance = value;
    }

}
